"""数据层：读 data/ 原始日线与合约表，建三腿同月换月日历，算每组合约自己的价差 S 与 z。

约定
- 一组合约 = 同一交割月的 L、PP、MA，用交割月 'YYYYMM' 标识
- 1 单位 = 1 手 L + 1 手 PP + 3 手 MA；S = 5·L + 5·PP − 30·MA（元/单位，每手吨数取 contracts 的 per_unit）
- μ、σ 只用「当天日历所指那组合约」自己过去 w 个交易日的 S，不含当日，不用拼接序列
"""
import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / 'data/daily_MA_L_PP.csv'
CONTRACTS = ROOT / 'data/contracts_MA_L_PP.csv'
LEGS = ('L', 'PP', 'MA')
LOTS = {'L': 1, 'PP': 1, 'MA': 3}
SIGN = {'L': 1, 'PP': 1, 'MA': -1}
# 合约表对所有 L 合约都写 1 元/吨（现行规则）；日线显示此日前有成交的 L 价格全是 5 的整数倍，
# 此日起不是，判断此前最小变动价位为 5 元/吨（证据见 check_data.tick_evidence 与 tests/test_data.py）
L_TICK5_BEFORE = '20211101'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw():
    daily = pd.read_csv(DAILY, dtype={'trade_date': str})
    contracts = pd.read_csv(CONTRACTS, dtype={'list_date': str, 'delist_date': str,
                                              'd_month': str, 'last_ddate': str})
    return daily, contracts


@dataclass
class Market:
    dates: np.ndarray   # 交易日 'YYYYMMDD'，升序
    months: list        # 三腿都有日线的交割月 'YYYYMM'，升序
    close: dict         # leg -> DataFrame(交易日 × 交割月)
    open: dict
    vol: dict
    oi: dict
    unit: dict          # leg -> 每手吨数
    tick: dict          # leg -> DataFrame 最小变动价位（元/吨），L 随时间变
    expiry: pd.Series   # 交割月 -> 三腿中最早的最后交易日
    codes: pd.DataFrame  # 交割月 × leg -> ts_code

    def value(self, px, signed=True):
        """按 1 单位把三腿价格合成：signed=True 得价差 S，False 得名义总额。"""
        return sum((SIGN[l] if signed else 1) * LOTS[l] * self.unit[l] * px[l] for l in LEGS)

    @cached_property
    def S_close(self):
        return self.value(self.close)

    @cached_property
    def S_open(self):
        return self.value(self.open)

    @cached_property
    def notional(self):
        return self.value(self.close, signed=False)


def build_market(daily, contracts):
    spec = contracts.set_index('ts_code')
    d = daily.join(spec[['fut_code', 'd_month']], on='ts_code')
    dates = np.sort(d['trade_date'].unique())
    months = sorted(set.intersection(*[set(d.loc[d.fut_code == l, 'd_month']) for l in LEGS]))
    panels = {f: {} for f in ('close', 'open', 'vol', 'oi')}
    unit, tick, codes, last = {}, {}, {}, {}
    for leg in LEGS:
        dl = d[d.fut_code == leg]
        grid = lambda f: dl.pivot(index='trade_date', columns='d_month', values=f).reindex(index=dates, columns=months)
        for f in panels:
            panels[f][leg] = grid(f)
        # 无成交日：收盘价缺失用当日结算价，开盘价缺失用补过的收盘价（任务 0 查清的数据规则）
        panels['close'][leg] = panels['close'][leg].fillna(grid('settle'))
        panels['open'][leg] = panels['open'][leg].fillna(panels['close'][leg])
        sl = spec[(spec.fut_code == leg) & spec.d_month.isin(months)]
        unit[leg] = sl['per_unit'].unique().item()
        t0 = float(re.match(r'[\d.]+', sl['quote_unit_desc'].unique().item()).group())
        tick[leg] = pd.DataFrame(t0, index=dates, columns=months)
        codes[leg] = sl.reset_index().set_index('d_month')['ts_code'].reindex(months)
        last[leg] = sl.set_index('d_month')['delist_date'].reindex(months)
    tick['L'].loc[dates < L_TICK5_BEFORE] = 5.0
    expiry = pd.concat(last, axis=1).min(axis=1)
    return Market(dates, months, panels['close'], panels['open'], panels['vol'], panels['oi'],
                  unit, tick, expiry, pd.DataFrame(codes))


def load_market():
    return build_market(*load_raw())


def prev_month_start(month):
    """交割月前一个月的 1 日，'YYYYMMDD'。"""
    return (pd.Timestamp(month + '01') - pd.DateOffset(months=1)).strftime('%Y%m%d')


def roll_calendar(mkt, buffer=3):
    """修正版换月日历：每天在持哪一组合约（交割月）。

    - 候选：前一交易日三腿都有收盘价、且今天不晚于「交割月前一个月首个交易日」前 buffer 个交易日的组
    - 选前一交易日 MA 持仓量最大的组（MA 主力），只向后换不回头
    - buffer=3 保证收盘成交与次日开盘成交两种口径都在交割月前一个月之前换完
    """
    dates = mkt.dates
    oi = mkt.oi['MA'].to_numpy()
    ok = np.logical_and.reduce([mkt.close[l].notna().to_numpy() for l in LEGS]) & ~np.isnan(oi)
    ftd_idx = np.array([np.searchsorted(dates, prev_month_start(m)) for m in mkt.months])
    last_ok = np.where(ftd_idx < len(dates), ftd_idx - buffer, len(dates))  # 截止日在数据之后的组到末尾都可选
    out = [None] * len(dates)
    cur = None
    for i in range(1, len(dates)):
        cand = np.flatnonzero(ok[i - 1] & (i <= last_ok))
        if cand.size == 0:
            cur = None
            continue
        main = cand[np.argmax(oi[i - 1, cand])]
        if cur is None or i > last_ok[cur] or main > cur:
            cur = main
        out[i] = mkt.months[cur]
    return pd.Series(out, index=dates, name='month', dtype=object)


def expiry_calendar(mkt, roll_cal, start):
    """原版（不换月）：从 start 当天换月日历所指的组开始，一组合约一直用到最后交易日，次日才换下一组，
    组的先后顺序与换月日历相同。start 之前沿用换月日历（只用于取评估首日前一天的名义额）。"""
    seq = list(dict.fromkeys(roll_cal.loc[start:].dropna()))
    out = roll_cal.copy()
    k = 0
    for day in roll_cal.loc[start:].index:
        while day > mkt.expiry[seq[k]]:
            k += 1
        out[day] = seq[k]
    return out


def holdings(cal):
    """每个交易日可能在持的交割月：当日与前两日的日历月份（覆盖收盘成交与次日开盘成交两种口径）。"""
    return pd.DataFrame({'d0': cal, 'd1': cal.shift(1), 'd2': cal.shift(2)})


def rolling_stats(S, w):
    """每组合约各自的滚动均值与标准差（np.std 口径 ddof=0），窗口为当日之前 w 个交易日。"""
    past = S.shift(1)
    return past.rolling(w).mean(), past.rolling(w).std(ddof=0)


def zscore(S, w):
    mu, sd = rolling_stats(S, w)
    return (S - mu) / sd


def pick(panel, cal):
    """取每天日历所指那组合约的值，返回按交易日索引的 Series。"""
    j = panel.columns.get_indexer(cal.fillna(''))
    v = panel.to_numpy()[np.arange(len(panel)), j]
    return pd.Series(np.where(j >= 0, v, np.nan), index=panel.index)
