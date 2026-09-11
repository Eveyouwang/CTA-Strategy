"""任务 1 验收：换月日历与滚动窗口。"""
import numpy as np
import pandas as pd
import pytest

from src.check_data import tick_evidence
from src.data import (L_TICK5_BEFORE, holdings, load_market, pick, prev_month_start, roll_calendar,
                      rolling_stats, zscore)

W = 29


@pytest.fixture(scope='module')
def mkt():
    return load_market()


@pytest.fixture(scope='module')
def cal(mkt):
    return roll_calendar(mkt)


def test_never_hold_delivery_month(cal):
    """① 任何一天可能在持的合约都不在交割月，并且更严：都在交割月前一个月首日之前。"""
    h = holdings(cal).dropna()
    day_month = h.index.str[:6]
    for c in h.columns:
        assert (day_month != h[c]).all()
        assert (h.index < h[c].map(prev_month_start)).all()


def test_window_excludes_today(mkt):
    """② μ、σ 只用当日之前 29 天：与手算一致；改动当日 S 不影响当日 μ、σ。"""
    S = mkt.S_close
    mu, sd = rolling_stats(S, W)
    valid = np.argwhere(mu.notna().to_numpy())
    rng = np.random.default_rng(0)
    for i, j in valid[rng.choice(len(valid), 300, replace=False)]:
        past = S.iloc[i - W:i, j].to_numpy()
        assert np.isclose(mu.iat[i, j], past.mean(), rtol=1e-12, atol=1e-6)
        assert np.isclose(sd.iat[i, j], np.std(past), rtol=1e-9, atol=1e-6)
    i, j = valid[len(valid) // 2]
    S2 = S.copy()
    S2.iat[i, j] += 1e6
    mu2, sd2 = rolling_stats(S2, W)
    assert mu2.iat[i, j] == mu.iat[i, j] and sd2.iat[i, j] == sd.iat[i, j]
    assert zscore(S2, W).iat[i, j] != zscore(S, W).iat[i, j]


def test_switch_day_uses_new_set_history(mkt, cal):
    """换月当天的 z 用新合约组自己过去 29 天的 S，而不是「旧组历史 + 新组当日」的拼接序列。"""
    S = mkt.S_close
    za = pick(zscore(S, W), cal)
    spliced = pick(S, cal)
    switch = cal.index[(cal != cal.shift()) & cal.shift().notna()]
    assert len(switch) > 20
    differs = 0
    for d in switch:
        i, m = mkt.dates.searchsorted(d), cal[d]
        past = S[m].iloc[i - W:i].to_numpy()
        if np.isnan(past).any():
            continue
        assert np.isclose(za[d], (S[m].iloc[i] - past.mean()) / np.std(past))
        sp = spliced.iloc[i - W:i].to_numpy()
        differs += not np.isclose((S[m].iloc[i] - sp.mean()) / np.std(sp), za[d])
    assert differs > 0


def test_calendar_same_month_forward_only(mkt, cal):
    """三腿同一交割月；日历只向后换；郑商所 ts_code 用 4 位年月（MA2409.ZCE）。"""
    held = cal.dropna()
    assert held.is_monotonic_increasing
    first = held.index[0]
    assert cal.loc[first:].notna().all()
    for m in held.unique():
        c = mkt.codes.loc[m]
        assert (c['L'], c['PP'], c['MA']) == (f'L{m[2:]}.DCE', f'PP{m[2:]}.DCE', f'MA{m[2:]}.ZCE')


def test_l_tick_follows_prices(mkt):
    """L 的最小变动价位：此日前有成交价格几乎全是 5 的整数倍（按 5 元计），此后按 1 元计。"""
    ev = tick_evidence()
    assert ev.loc['L', '之前'] > 0.99 and ev.loc['L', '之后'] < 0.1
    assert ev.loc[['PP', 'MA'], '之前'].max() < 0.2
    t = mkt.tick['L']
    assert (t[t.index < L_TICK5_BEFORE] == 5).all().all() and (t[t.index >= L_TICK5_BEFORE] == 1).all().all()
