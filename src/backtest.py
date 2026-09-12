"""回测引擎：每日收盘算信号；成交价两套——乐观 fill='close' 为信号日收盘，保守 fill='open' 为次日开盘。

- 仓位 pos ∈ {-1, 0, 1}：-1 做空利润（卖 L、卖 PP、买 MA），1 做多利润，1 单位 = 1 手 L + 1 手 PP + 3 手 MA
- 成本：每腿每边 手续费 = 名义额 × 0.0001 + 滑点 ticks 跳；换月（平旧开新）同样计
- 合约到期日（原版不换月时会碰到）与样本最后一天：收盘强平
- 日收益率 = 当日净盈亏 / (12% × 前一交易日在持那组合约 1 单位名义总额)，按日简单累加成净值
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import LEGS, LOTS, pick

FEE = 1e-4
MARGIN = 0.12
DAYS = 252


@dataclass(frozen=True)
class Rule:
    entry: float = 2.0              # z 超过 ±entry 开仓
    exit: float = 0.5               # 平仓阈值
    stop_mult: float = 1.5          # 止损阈值 = entry × 1.5（原版 STD_THRESHOLD * 1.5 = 3）
    cooldown: bool = False          # 修正①：止损后 |z| 回到 entry 以内才可再开
    directional_exit: bool = False  # 修正④：空头 z<exit、多头 z>-exit 即平；原版为 |z|<exit
    money_stop: float = 0.0         # 第三轮风控：持仓浮亏达到该仓位保证金的这个比例即平，0 为不启用


def decide(pos, z, rule, cool):
    """原版信号逻辑。返回 (新仓位, 原因, 冷静期标记)。"""
    if np.isnan(z):
        return pos, None, cool
    if pos == 0:
        if cool and abs(z) < rule.entry:
            cool = False
        if not cool and z > rule.entry:
            return -1, 'open', cool
        if not cool and z < -rule.entry:
            return 1, 'open', cool
        return 0, None, cool
    if (pos * z > -rule.exit) if rule.directional_exit else (abs(z) < rule.exit):
        return 0, 'take_profit', cool
    if pos * z < -rule.entry * rule.stop_mult:
        return 0, 'stop_loss', rule.cooldown
    return pos, None, cool


def unit_cost(mkt, px, ticks):
    """1 单位单边成本（元）= Σ 手数 × 每手吨数 × (价格 × 费率 + 跳数 × 最小变动价位)。"""
    return sum(LOTS[l] * mkt.unit[l] * (px[l] * FEE + ticks * mkt.tick[l]) for l in LEGS)


def backtest(mkt, cal, z, rule, fill='close', ticks=1, start=None, end=None, size=None, capital=1):
    """cal：每日在持交割月；z：交易日 × 交割月 的 z 面板。
    size：每日手数（按交易日索引的 Series），开仓当天取值并在持仓期间不变，None 为固定 1 单位；
    capital：本金相当于几倍 1 单位保证金（只影响收益率口径，不影响信号与成交）。返回 (逐日表, 交易表)。"""
    dates = mkt.dates
    i0 = int(np.searchsorted(dates, start))
    i1 = int(np.searchsorted(dates, end, side='right')) - 1
    col = {m: j for j, m in enumerate(mkt.months)}
    mon = cal.reindex(dates).to_numpy()
    assert all(isinstance(m, str) for m in mon[i0 - 1:i1 + 1]), '评估期内每天都要有在持合约组'
    Z = z.to_numpy()
    Sc, So = mkt.S_close.to_numpy(), mkt.S_open.to_numpy()
    Cc = unit_cost(mkt, mkt.close, ticks).to_numpy()
    Co = unit_cost(mkt, mkt.open, ticks).to_numpy()
    N = mkt.notional.to_numpy()
    base = capital * MARGIN * pick(mkt.notional, cal.reindex(dates)).shift(1).to_numpy()
    qty = np.ones(len(dates)) if size is None else size.reindex(dates).to_numpy()
    assert np.isfinite(qty[i0:i1 + 1]).all(), '评估期内每天都要有手数'
    exp_i = {m: int(np.searchsorted(dates, mkt.expiry[m])) for m in set(mon[i0 - 1:i1 + 1])}

    pos, held, cool, pending, cur, q = 0, None, False, None, None, 0.0
    rows, trades = [], []

    def mark(x):  # 逐段盯市，记入当前交易；mae 为持仓期间累计毛盈亏的最低点
        cur['pnl'] += x
        cur['mae'] = min(cur['mae'], cur['pnl'])
        return x

    def execute(i, when, new, month, sig, reason):
        nonlocal pos, held, cur, q
        if new == pos and month == held:
            return 0.0
        C, S = (Cc, Sc) if when == 'close' else (Co, So)
        if pos and new == pos:  # 换月：同方向移仓，手数不变，交易记录延续
            out = (C[i, col[held]] + C[i, col[month]]) * q
            cur['rolls'] += 1
            cur['cost'] += out
        else:
            out = C[i, col[held]] * q if pos else 0.0
            if pos:
                cur.update(close_signal=dates[sig], close_exec=dates[i], reason=reason,
                           z_close=Z[sig, col[mon[sig]]], S_exit=S[i, col[held]],
                           hold_days=i - cur.pop('_i'), cost=cur['cost'] + C[i, col[held]] * q)
                trades.append(cur)
                cur, q = None, 0.0
            if new:
                q = qty[sig]
                out += C[i, col[month]] * q
                cur = dict(open_signal=dates[sig], open_exec=dates[i], side=new, month=month,
                           z_open=Z[sig, col[month]], S_entry=S[i, col[month]], _i=i, qty=q,
                           margin=MARGIN * N[sig, col[month]] * q, rolls=0, pnl=0.0, mae=0.0,
                           cost=C[i, col[month]] * q)
        pos, held = new, month
        return out

    for i in range(i0, i1 + 1):
        m = mon[i]
        pnl = cost = 0.0
        if fill == 'close':
            if pos:
                pnl = mark(q * pos * (Sc[i, col[held]] - Sc[i - 1, col[held]]))
        else:
            if pos:  # 隔夜：前一日收盘到今日开盘，仍是旧仓位
                pnl = mark(q * pos * (So[i, col[held]] - Sc[i - 1, col[held]]))
            if pending:
                cost += execute(i, 'open', *pending)
                pending = None
            if pos:  # 日内：今日开盘到收盘，成交后的仓位
                pnl += mark(q * pos * (Sc[i, col[held]] - So[i, col[held]]))
        new, reason, cool = decide(pos, Z[i, col[m]], rule, cool)
        if pos and new == pos and rule.money_stop and cur['pnl'] < -rule.money_stop * cur['margin']:
            new, reason = 0, 'money_stop'  # 金额止损：浮亏超过该仓位保证金的给定比例
        if pos and (i == i1 or i == exp_i[held]):
            new, reason = 0, ('end' if i == i1 else 'expiry')
        elif not pos and (i == i1 or i == exp_i[m]):  # 最后一天、到期日不开新仓
            new, reason = 0, None
        if fill == 'close' or reason in ('end', 'expiry'):
            cost += execute(i, 'close', new, m, i, reason)
        else:
            pending = (new, m, i, reason)
        rows.append((dates[i], pnl, cost, pos, held, base[i]))

    assert cur is None
    daily = pd.DataFrame(rows, columns=['date', 'pnl', 'cost', 'pos', 'month', 'base']).set_index('date')
    assert daily[['pnl', 'cost', 'base']].notna().all().all(), '在持合约有缺价，盈亏算不出'
    daily['ret'] = (daily['pnl'] - daily['cost']) / daily['base']
    cols = ['open_signal', 'open_exec', 'close_signal', 'close_exec', 'side', 'month', 'reason', 'qty',
            'z_open', 'z_close', 'S_entry', 'S_exit', 'hold_days', 'rolls', 'pnl', 'mae', 'cost', 'margin']
    t = pd.DataFrame(trades, columns=cols)
    t['net'] = t['pnl'] - t['cost']
    t['ret'] = t['net'] / t['margin']
    return daily, t


def metrics(daily, trades):
    r = daily['ret']
    nav = 1 + r.cumsum()
    vol = r.std() * np.sqrt(DAYS)
    return {
        'ann_ret': r.mean() * DAYS,
        'mdd': (1 - nav / nav.cummax().clip(lower=1)).max(),
        'sharpe': r.mean() * DAYS / vol if vol > 0 else np.nan,
        'win': (trades['net'] > 0).mean() if len(trades) else np.nan,
        'n': len(trades),
        'hold': trades['hold_days'].mean() if len(trades) else np.nan,
        'n_stop': int((trades['reason'] == 'stop_loss').sum()),
        'n_roll': int(trades['rolls'].sum()),
        'cost_share': daily['cost'].sum() / daily['base'].mean(),  # 累计成本占平均保证金
    }
