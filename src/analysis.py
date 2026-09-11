"""任务 2–5 的计算：原版复现、逐项修正、稳健性、样本外。报告里的每个数都从这里来。

两轮设计（详见 PROGRESS.md「第二轮」）：
- 第一轮：原始设计，样本内 2015-05-04 起；它的样本外是唯一一次干净的检验，结果原样保留
- 第二轮：看过第一轮样本外之后按用户要求修订——样本内改从三腿交易时段一致的 2019-12-12 起、
  换月只在 01/05/09 里选、③窗口按跑之前写定的规则；2025-07-01 起的样本外已被看过，只作参考

python -m src.analysis select round2   只用第二轮样本内数据跑网格、按邻域稳定选参，写 report/round2/params.json
"""
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .backtest import DAYS, MARGIN, Rule, backtest, metrics, unit_cost
from .data import (LEGS, LOTS, ROOT, expiry_calendar, load_market, pick, prev_month_start, roll_calendar,
                   rolling_stats, zscore)

REPORT = ROOT / 'report'
DEMO = dict(month='202409', start='20231101', end='20240430')  # 原版 demo 写死的合约与回测区间
WINDOWS, ENTRIES, EXITS = (20, 30, 60, 120), (1.5, 2.0, 2.5), (0.0, 0.5, 1.0)
FILLS = ('close', 'open')
W0 = 29  # 原版：30 根日线去掉当根
MAIN_MONTHS = ('01', '05', '09')


@dataclass(frozen=True)
class Round:
    name: str
    is_start: str
    is_end: str
    oos_start: str
    months: tuple | None  # 换月候选交割月，None 为全部月份
    window_rule: str      # ③ 窗口规则，见 window3

    @property
    def out(self):
        return REPORT / self.name

    @property
    def params(self):
        return self.out / 'params.json'


# 第一轮起点按流动性定：2015-04 之前在持的 1506 组 L、PP 腿日成交只有几手到几十手
ROUND1 = Round('round1', '20150504', '20250630', '20250701', None, 'grid_le_hl')
# 第二轮起点为三腿交易时段完全一致的首个交易日：大商所 L、PP 自 2019-03-29 起有夜盘（21:00–23:00），
# 郑商所夜盘自 2019-12-11 晚起统一到 23:00 结束
ROUND2 = Round('round2', '20191212', '20250630', '20250701', MAIN_MONTHS, 'hl_capped_by_history')
ROUNDS = {r.name: r for r in (ROUND1, ROUND2)}


class Lab:
    def __init__(self, r, mkt=None):
        self.r = r
        self.mkt = load_market() if mkt is None else mkt
        roll = roll_calendar(self.mkt, months=r.months)
        self.cal = {'roll': roll, 'expiry': expiry_calendar(self.mkt, roll, r.is_start)}
        self._z = {}

    def z(self, w):
        if w not in self._z:
            self._z[w] = zscore(self.mkt.S_close, w)
        return self._z[w]

    def run(self, cal, w, rule, fill, ticks, start, end):
        c = self.cal[cal] if isinstance(cal, str) else cal
        return backtest(self.mkt, c, self.z(w), rule, fill=fill, ticks=ticks, start=start, end=end)


# ---------- 任务 2 原版 ----------
def original(lab):
    """原版逻辑：窗口 29、z>2 空 / z<-2 多、|z|<0.5 平、反向超 3 止损、无冷静期、不换月。
    demo_2409：原脚本写死的 2409 合约与 2023-11-01~2024-04-30；full_is：同一逻辑放到整个样本内，一组合约用到最后交易日。"""
    demo_cal = pd.Series(DEMO['month'], index=lab.mkt.dates, dtype=object)
    scopes = {'demo_2409': (demo_cal, DEMO['start'], DEMO['end']),
              'full_is': ('expiry', lab.r.is_start, lab.r.is_end)}
    rows, trades, daily = [], [], {}
    for scope, (cal, s, e) in scopes.items():
        for fill in FILLS:
            for ticks in (0, 1):
                d, t = lab.run(cal, W0, Rule(), fill, ticks, s, e)
                rows.append(dict(scope=scope, fill=fill, ticks=ticks, **metrics(d, t)))
                trades.append(t.assign(scope=scope, fill=fill, ticks=ticks))
                daily[scope, fill, ticks] = d
    return pd.DataFrame(rows), pd.concat(trades, ignore_index=True), daily


def defects(lab, daily, trades):
    """用原版全样本内（1 跳）的结果给缺陷定量。"""
    mkt, dates = lab.mkt, lab.mkt.dates
    out = {}
    for fill in FILLS:
        t = trades[(trades.scope == 'full_is') & (trades.fill == fill) & (trades.ticks == 1)].reset_index(drop=True)
        d = daily['full_is', fill, 1]
        gap = dates.searchsorted(t.open_signal.to_numpy()[1:]) - dates.searchsorted(t.close_signal.to_numpy()[:-1])
        hit = (gap == 1) & (t.reason.to_numpy()[:-1] == 'stop_loss') & (t.side.to_numpy()[1:] == t.side.to_numpy()[:-1])
        re = t.iloc[1:][hit]
        held = d[d.pos != 0]
        late = held.index.to_numpy() >= held.month.map(prev_month_start).to_numpy()
        vol = pd.Series(mkt.vol['MA'].to_numpy()[dates.searchsorted(held.index),
                                                  [mkt.months.index(m) for m in held.month]], index=held.index)
        za = pick(lab.z(W0), lab.cal['expiry'])
        over = [((za.loc[r.open_signal:r.close_signal].iloc[1:] * r.side) > 0.5).any() for r in t.itertuples()]
        out[fill] = dict(
            n_trades=len(t), n_stop=int((t.reason == 'stop_loss').sum()), n_reentry=len(re),
            reentry_net=re.net.sum(), reentry_win=(re.net > 0).mean() if len(re) else np.nan,
            n_expiry=int((t.reason == 'expiry').sum()), held_days=len(held), late_days=int(late.sum()),
            deliv_days=int((held.index.str[:6] == held.month).sum()),
            late_net=(held.pnl - held.cost)[late].sum(), total_net=(d.pnl - d.cost).sum(),
            vol_late=vol[late].median(), vol_normal=vol[~late].median(),
            n_overshoot=int(np.sum(over)), overshoot_net=t.net[np.array(over, bool)].sum())
    return out


# ---------- 任务 3 逐项修正 ----------
def versions(w3):
    """在原版上逐项累加，每步只改一处。"""
    v0 = dict(cal='expiry', w=W0, rule=Rule())
    v1 = dict(v0, rule=Rule(cooldown=True))
    v2 = dict(v1, cal='roll')
    v3 = dict(v2, w=w3)
    v4 = dict(v3, rule=Rule(cooldown=True, directional_exit=True))
    return [('v0 原版', v0), ('v1 +①止损冷静期', v1), ('v2 +②换月', v2),
            (f'v3 +③窗口{w3}天', v3), ('v4 +④方向性平仓', v4)]


def fixes(lab, w3):
    rows, daily = [], {}
    for name, v in versions(w3):
        for fill in FILLS:
            d, t = lab.run(v['cal'], v['w'], v['rule'], fill, 1, lab.r.is_start, lab.r.is_end)
            rows.append(dict(version=name, fill=fill, **metrics(d, t)))
            daily[name, fill] = d
    return pd.DataFrame(rows), daily


def history_limit(lab):
    """样本内每天在持组自有历史（当日之前连续有效 S 的天数）的最小值：窗口不超过它，z 每天都算得出。"""
    valid = lab.mkt.S_close.notna()
    c = valid.cumsum()
    run = c - c.where(~valid).ffill().fillna(0)
    return int(pick(run.shift(1), lab.cal['roll']).loc[lab.r.is_start:lab.r.is_end].min())


def window3(lab, hl):
    """③ 的窗口。第一轮（看过样本内网格后改的）：不超过半衰期的最大网格窗口。
    第二轮（跑之前写定）：半衰期四舍五入；在持组自有历史不够时，取样本内每天都算得出 z 的最长窗口。"""
    if lab.r.window_rule == 'grid_le_hl':
        return max(w for w in WINDOWS if w <= hl)
    return min(int(round(hl)) if np.isfinite(hl) else 10 ** 6, history_limit(lab))


def z_coverage(lab, w):
    """换月日历下样本内 z 能算出的天数、总天数。"""
    za = pick(lab.z(w), lab.cal['roll']).loc[lab.r.is_start:lab.r.is_end]
    return int(za.notna().sum()), len(za)


# ---------- 任务 4 稳健性（只用样本内）----------
def rule_of(entry, exit_, stop_mult=1.5):
    return Rule(entry=entry, exit=exit_, stop_mult=stop_mult, cooldown=True, directional_exit=True)


def grid(lab):
    rows = []
    for w in WINDOWS:
        for e in ENTRIES:
            for x in EXITS:
                for fill in FILLS:
                    d, t = lab.run('roll', w, rule_of(e, x), fill, 1, lab.r.is_start, lab.r.is_end)
                    rows.append(dict(w=w, entry=e, exit=x, fill=fill, **metrics(d, t)))
    return pd.DataFrame(rows)


def select(g, r):
    """邻域稳定选参：每格与其在 窗口/开仓/平仓 三个方向上相邻格（含对角，边界处截断）的
    保守口径（次日开盘、1 跳）夏普取平均，选平均最高的格；无交易的格夏普记 0。"""
    s = g[g.fill == 'open'].set_index(['w', 'entry', 'exit'])['sharpe'].sort_index().fillna(0)
    a = s.to_numpy().reshape(len(WINDOWS), len(ENTRIES), len(EXITS))
    nb = np.empty_like(a)
    for i, j, k in np.ndindex(a.shape):
        nb[i, j, k] = a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2, max(k - 1, 0):k + 2].mean()
    i, j, k = np.unravel_index(np.argmax(nb), a.shape)
    bi, bj, bk = np.unravel_index(np.argmax(a), a.shape)
    return dict(window=WINDOWS[i], entry=ENTRIES[j], exit=EXITS[k], stop_mult=1.5, cooldown=True,
                directional_exit=True, calendar='roll', buffer=3, months=r.months, round=r.name,
                sharpe=float(a[i, j, k]), neighborhood_sharpe=float(nb[i, j, k]),
                best_single=dict(window=WINDOWS[bi], entry=ENTRIES[bj], exit=EXITS[bk],
                                 sharpe=float(a[bi, bj, bk]), neighborhood_sharpe=float(nb[bi, bj, bk])),
                sample=[r.is_start, r.is_end], neighborhood=nb.tolist())


def yearly(daily, trades):
    r = daily['ret']
    g = r.groupby(r.index.str[:4])
    nav_dd = lambda s: (1 - (1 + s.cumsum()) / (1 + s.cumsum()).cummax().clip(lower=1)).max()
    out = pd.DataFrame({'ret': g.sum(), 'sharpe': g.mean() / g.std() * np.sqrt(DAYS), 'mdd': g.apply(nav_dd)})
    out['n'] = trades.groupby(trades.open_exec.str[:4]).size().reindex(out.index).fillna(0).astype(int)
    return out


def half_life(x):
    """AR(1)：Δx_t = a + b·x_{t-1}，半衰期 = -ln2 / ln(1+b)；b>=0 时不回归，记为无穷。"""
    x = np.asarray(x, float)
    b = np.linalg.lstsq(np.column_stack([np.ones(len(x) - 1), x[:-1]]), np.diff(x), rcond=None)[0][1]
    return -np.log(2) / np.log(1 + b) if b < 0 else np.inf


def stationarity(lab):
    """ADF 与半衰期：①在持组逐日价差变化累加成的连续序列（换月日取旧组当天变化，去掉换月跳空）；
    ②每组合约在其在持期内自己的 S。"""
    start, end = lab.r.is_start, lab.r.is_end
    S, cal = lab.mkt.S_close, lab.cal['roll']
    prev = cal.shift(1)
    adj = (pick(S, prev) - pick(S.shift(1), prev)).loc[start:end].cumsum()
    stat, p, lags, nobs = adfuller(adj, autolag='AIC')[:4]
    c = cal.loc[start:end]
    per = [(m, len(ix), adfuller(S.loc[ix, m], autolag='AIC')[1], half_life(S.loc[ix, m]))
           for m, ix in c.groupby(c).groups.items() if len(ix) >= 40]
    per = pd.DataFrame(per, columns=['month', 'n', 'p', 'hl'])
    return dict(adj=adj, adj_stat=stat, adj_p=p, adj_lags=lags, adj_n=nobs, adj_hl=half_life(adj), per=per)


def costs(lab, params, start, end):
    rule = rule_of(params['entry'], params['exit'], params['stop_mult'])
    rows, runs = [], {}
    for fill in FILLS:
        for ticks in (0, 1, 2):
            d, t = lab.run('roll', params['window'], rule, fill, ticks, start, end)
            rows.append(dict(fill=fill, ticks=ticks, **metrics(d, t)))
            runs[fill, ticks] = (d, t)
    return pd.DataFrame(rows), runs


# ---------- 换月记录与实盘视角 ----------
def roll_stats(lab):
    """换月记录（全数据区间）；样本内换月次数、换月日距「交割月前一个月首日」的交易日数、换到非 01/05/09 的次数。"""
    dates, cal, r = lab.mkt.dates, lab.cal['roll'], lab.r
    sw = cal.index[(cal != cal.shift()) & cal.shift().notna()]
    old = cal.shift(1).loc[sw]
    lead = np.array([dates.searchsorted(prev_month_start(m)) - dates.searchsorted(d) for d, m in zip(sw, old)])
    tab = pd.DataFrame({'date': sw, 'from': old.to_numpy(), 'to': cal.loc[sw].to_numpy(), 'lead_days': lead})
    tab = tab.join(lab.mkt.codes.add_prefix('to_'), on='to')
    ins = (tab.date >= r.is_start) & (tab.date <= r.is_end)
    odd = ins & ~tab['to'].str[4:].isin(MAIN_MONTHS)
    lag = dates.searchsorted(tab.date.shift(-1)[odd]) - dates.searchsorted(tab.date[odd])
    return dict(n_is=int(ins.sum()), lead_med=float(np.median(lead[ins])), table=tab, n_odd=int(odd.sum()),
                lag=(int(lag.min()), int(lag.max())) if odd.any() else None)


def term_structure(lab):
    """换月日新组与旧组同日收盘 S 之差（新 − 旧），以及在持组原始 S 的首末值。"""
    start, end = lab.r.is_start, lab.r.is_end
    S, cal = lab.mkt.S_close, lab.cal['roll']
    c = cal.loc[start:end]
    sw = c.index[(c != c.shift()) & c.shift().notna()]
    gap = pd.Series([S.at[d, n] - S.at[d, o] for d, o, n in zip(sw, cal.shift(1).loc[sw], c.loc[sw])], index=sw)
    raw = pick(S, cal).loc[start:end]
    return dict(n=len(gap), gap_mean=gap.mean(), gap_neg=(gap < 0).mean(), raw0=raw.iloc[0], raw1=raw.iloc[-1])


def live(lab, runs):
    """隔夜跳空、名义额与保证金、单边成本、持仓期最大浮亏、在持合约成交量。"""
    start, end = lab.r.is_start, lab.r.is_end
    mkt, cal = lab.mkt, lab.cal['roll']
    gap = (pick(mkt.S_open, cal) - pick(mkt.S_close.shift(1), cal)).loc[start:end].abs()
    sd = pick(rolling_stats(mkt.S_close, W0)[1], cal).loc[start:end]
    notional = pick(mkt.notional, cal).loc[start:end]
    vol = {l: pick(mkt.vol[l], cal).loc[start:end] for l in LEGS}
    c = cal.loc[start:end]
    sw = c.index[(c != c.shift()) & c.shift().notna()]
    t = runs['open', 1][1]
    return dict(gap_med=gap.median(), gap_p90=gap.quantile(0.9), sd_med=sd.median(),
                notional_med=notional.median(), margin_med=MARGIN * notional.median(),
                cost_side_med=pick(unit_cost(mkt, mkt.close, 1), cal).loc[start:end].median(),
                slip_before=sum(LOTS[l] * mkt.unit[l] * mkt.tick[l].loc[start].iloc[0] for l in LEGS),
                slip_after=sum(LOTS[l] * mkt.unit[l] * mkt.tick[l].loc[end].iloc[0] for l in LEGS),
                mae_med=(t.mae / t.margin).median(), mae_worst=(t.mae / t.margin).min(),
                vol_med={l: v.median() for l, v in vol.items()},
                vol_roll={l: v.loc[sw].median() for l, v in vol.items()},
                demo_share={l: q / vol[l].median() for l, q in (('L', 100), ('PP', 100), ('MA', 300))})


def main():
    assert sys.argv[1:] == ['select', 'round2'], '用法：python -m src.analysis select round2（第一轮参数已冻结，不重选）'
    r = ROUND2
    p = select(grid(Lab(r)), r)
    r.out.mkdir(parents=True, exist_ok=True)
    r.params.write_text(json.dumps(p, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in p.items() if k != 'neighborhood'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
