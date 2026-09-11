"""回测引擎验收：信号逻辑逐条、两套成交价、成本、冷静期、账目恒等、不持有交割月。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import Rule, backtest, unit_cost
from src.data import LEGS, Market, load_market, prev_month_start, roll_calendar, zscore

#       p: 0  1    2    3    4  5     6     7     8     9     10   11
Z_PATH = [0, 2.5, 1.0, 0.3, 0, -2.5, -3.5, -2.5, -1.5, -2.5, 0.2, 0.0]


def toy(z_path, seed=1):
    """一组合约的小样本；第 0 天只做预热，z_path[p] 对应 dates[p + 1]。"""
    n = len(z_path) + 1
    dates = np.array([str(20200101 + k) for k in range(n)])
    rng = np.random.default_rng(seed)
    close, opn = {}, {}
    for leg, p0 in (('L', 8000), ('PP', 7500), ('MA', 2500)):
        c = p0 + rng.integers(-60, 60, n).cumsum()
        close[leg] = pd.DataFrame({'202012': c.astype(float)}, index=dates)
        opn[leg] = pd.DataFrame({'202012': (c + rng.integers(-20, 20, n)).astype(float)}, index=dates)
    one = {leg: close[leg] * 0 + 1 for leg in LEGS}
    mkt = Market(dates, ['202012'], close, opn, one, one, {'L': 5.0, 'PP': 5.0, 'MA': 10.0},
                 {'L': 1.0, 'PP': 1.0, 'MA': 1.0}, pd.Series({'202012': '20201215'}),
                 pd.DataFrame({'L': ['L2012.DCE'], 'PP': ['PP2012.DCE'], 'MA': ['MA2012.ZCE']},
                              index=['202012']))
    z = pd.DataFrame({'202012': [np.nan] + list(z_path)}, index=dates)
    cal = pd.Series('202012', index=dates, dtype=object)
    return mkt, cal, z


def run(z_path, rule, fill, ticks=0):
    mkt, cal, z = toy(z_path)
    daily, t = backtest(mkt, cal, z, rule, fill=fill, ticks=ticks, start=mkt.dates[1], end=mkt.dates[-1])
    return mkt, daily, t


@pytest.mark.parametrize('fill,lag', [('close', 0), ('open', 1)])
@pytest.mark.parametrize('cooldown,expect', [
    (False, [(1, 3, -1, 'take_profit'), (5, 6, 1, 'stop_loss'), (7, 10, 1, 'take_profit')]),
    (True, [(1, 3, -1, 'take_profit'), (5, 6, 1, 'stop_loss'), (9, 10, 1, 'take_profit')]),
])
def test_signal_logic_and_fills(fill, lag, cooldown, expect):
    """z>2 做空、z<-2 做多、|z|<0.5 平、反向超 3 止损；原版止损次日立刻同向重开，冷静期要等 |z|<2。
    乐观口径信号日收盘成交，保守口径次日开盘成交；毛盈亏 = 方向 × (出场 S − 入场 S)。"""
    mkt, daily, t = run(Z_PATH, Rule(cooldown=cooldown), fill)
    d = mkt.dates
    S = (mkt.S_close if fill == 'close' else mkt.S_open)['202012']
    got = list(zip(t.open_signal, t.close_signal, t.side, t.reason))
    assert got == [(d[p + 1], d[q + 1], s, r) for p, q, s, r in expect]
    assert list(t.open_exec) == [d[p + 1 + lag] for p, _, _, _ in expect]
    assert list(t.close_exec) == [d[q + 1 + lag] for _, q, _, _ in expect]
    for row in t.itertuples():
        assert np.isclose(row.pnl, row.side * (S[row.close_exec] - S[row.open_exec]))
        assert row.S_entry == S[row.open_exec] and row.S_exit == S[row.close_exec]
    assert np.isclose(t.net.sum(), (daily.pnl - daily.cost).sum())


def test_directional_exit():
    """原版 |z|<0.5 才平：做空后 z 直接穿到 -1 仍持有；方向性平仓在 z<0.5 时就平。"""
    path = [0, 2.5, -1.0, -1.0, 0.0, 0.0]
    _, _, t0 = run(path, Rule(), 'close')
    _, _, t1 = run(path, Rule(directional_exit=True), 'close')
    assert list(t0.reason) == ['take_profit'] and t0.close_signal.iloc[0] == '20200106'
    assert list(t1.reason) == ['take_profit'] and t1.close_signal.iloc[0] == '20200104'


@pytest.mark.parametrize('fill', ['close', 'open'])
def test_costs_per_leg_per_side(fill):
    """每腿每边：名义额 × 0.0001 + 滑点 n 跳；1 单位滑点 1 跳 = (5 + 5 + 30) 元。"""
    mkt, _, t0 = run(Z_PATH, Rule(), fill, ticks=0)
    _, _, t2 = run(Z_PATH, Rule(), fill, ticks=2)
    px = mkt.close if fill == 'close' else mkt.open
    fee = unit_cost(mkt, px, 0)['202012']
    for row in t0.itertuples():
        assert np.isclose(row.cost, fee[row.open_exec] + fee[row.close_exec])
    assert np.allclose(t2.cost - t0.cost, 2 * 2 * (5 + 5 + 30))


@pytest.fixture(scope='module')
def real():
    mkt = load_market()
    return mkt, roll_calendar(mkt), zscore(mkt.S_close, 29)


@pytest.mark.parametrize('fill', ['close', 'open'])
def test_real_run_accounting_and_no_delivery_month(real, fill):
    """真实数据：交易净盈亏合计 = 逐日净盈亏合计；有仓位的每一天都不在交割月（也不在其前一个月）。"""
    mkt, cal, z = real
    daily, t = backtest(mkt, cal, z, Rule(cooldown=True), fill=fill, ticks=1,
                        start='20150601', end='20250630')
    assert len(t) > 10 and t.rolls.sum() > 0
    assert np.isclose(t.net.sum(), (daily.pnl - daily.cost).sum())
    held = daily[daily.pos != 0]
    assert (held.index.str[:6] != held.month).all()
    assert (held.index < held.month.map(prev_month_start)).all()
