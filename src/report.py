"""生成 report/fig/*.png 与 report/report.md。数字全部取自 run_all.py 算出的结果 R，不手填。"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import DEMO, ENTRIES, EXITS, FILLS, IS_END, IS_START, OOS_START, REPORT, W0, WINDOWS  # noqa: E402
from .backtest import FEE, MARGIN  # noqa: E402
from .data import pick  # noqa: E402

FIG = REPORT / 'fig'
FILL_CN = {'close': '乐观(信号日收盘)', 'open': '保守(次日开盘)'}
FILL_EN = {'close': 'close fill', 'open': 'next-open fill'}
COLS = {'ann_ret': '年化收益', 'mdd': '最大回撤', 'sharpe': '夏普', 'win': '胜率', 'n': '交易次数',
        'hold': '平均持仓天数', 'n_stop': '止损次数', 'n_roll': '换月次数'}


def pct(v, d=1):
    return '—' if not np.isfinite(v) else f'{v * 100:.{d}f}%'


def num(v, d=2):
    return '—' if not np.isfinite(v) else f'{v:.{d}f}'


def day(s):
    return f'{s[:4]}-{s[4:6]}-{s[6:]}'


FMT = {'年化收益': pct, '最大回撤': pct, '夏普': num, '胜率': pct, '交易次数': lambda v: f'{int(v)}',
       '平均持仓天数': lambda v: num(v, 1), '止损次数': lambda v: f'{int(v)}', '换月次数': lambda v: f'{int(v)}'}


def table(df, fmt=None):
    fmt = {**FMT, **(fmt or {})}
    head = [df.index.name or ''] + [str(c) for c in df.columns]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for idx, row in df.iterrows():
        out.append('| ' + ' | '.join([str(idx)] + [fmt.get(c, str)(v) for c, v in row.items()]) + ' |')
    return '\n'.join(out)


def mtable(df, keys, name):
    t = df.copy()
    t.index = t[keys].astype(str).agg(' / '.join, axis=1)
    t.index.name = name
    return table(t[list(COLS)].rename(columns=COLS))


def row(df, **kw):
    m = np.logical_and.reduce([df[k] == v for k, v in kw.items()])
    return df[m].iloc[0]


def nav(d):
    return 1 + d['ret'].cumsum()


def up_down(a, b):
    return f"{'升' if b > a else '降'}至"


# ---------- 图 ----------
def fig_lines(panels, path, height=3.2):
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, height * len(panels)), squeeze=False)
    for ax, (title, curves, hlines) in zip(axes[:, 0], panels):
        for label, s in curves.items():
            ax.plot(pd.to_datetime(s.index), s.to_numpy(), label=label, lw=0.9)
        for h in hlines:
            ax.axhline(h, color='grey', lw=0.6, ls='--')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_heatmap(g, sel, path):
    vmax = np.nanmax(np.abs(g['sharpe']))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for r, fill in enumerate(FILLS):
        for c, x in enumerate(EXITS):
            ax = axes[r, c]
            a = (g[(g.fill == fill) & (g.exit == x)].pivot(index='w', columns='entry', values='sharpe')
                 .reindex(index=list(WINDOWS), columns=list(ENTRIES)).to_numpy())
            im = ax.imshow(a, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
            for (i, j), v in np.ndenumerate(a):
                ax.text(j, i, num(v), ha='center', va='center', fontsize=9)
            if fill == 'open' and x == sel['exit']:
                i, j = WINDOWS.index(sel['window']), ENTRIES.index(sel['entry'])
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='black', lw=2.5))
            ax.set_xticks(range(len(ENTRIES)), [f'{e:g}' for e in ENTRIES])
            ax.set_yticks(range(len(WINDOWS)), [str(w) for w in WINDOWS])
            ax.set_xlabel('entry |z|')
            ax.set_ylabel('window (days)')
            ax.set_title(f'{FILL_EN[fill]}, exit={x:g}', fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label='Sharpe (in-sample, 1 tick)')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def figures(R):
    FIG.mkdir(parents=True, exist_ok=True)
    lab, stat = R['lab'], R['stat']
    cal = lab.cal['roll']
    Sa = pick(lab.mkt.S_close, cal).loc[IS_START:]
    za = pick(lab.z(W0), cal).loc[IS_START:]
    fig_lines([('S of the held set (raw, yuan per unit; jumps at rolls)', {'S held set': Sa}, [0]),
               ('S with roll gaps removed (in-sample, used for ADF / half-life)', {'S adjusted': stat['adj']}, []),
               (f'z of the held set, window {W0}', {'z': za}, [-3, -2, -0.5, 0.5, 2, 3])],
              FIG / 'spread_z.png', 2.8)
    od = R['orig'][2]
    fig_lines([('Original logic, in-sample, fixed contract until expiry (1 tick)',
                {FILL_EN[f]: nav(od['full_is', f, 1]) for f in FILLS}, [1]),
               ('Original logic on the demo contracts 2409, 2023-11-01 to 2024-04-30 (1 tick)',
                {FILL_EN[f]: nav(od['demo_2409', f, 1]) for f in FILLS}, [1])],
              FIG / 'equity_original.png')
    en = ['v0 original', 'v1 +cooldown', 'v2 +roll', 'v3 +window', 'v4 +directional exit']
    names = R['fix'][0]['version'].unique()
    fig_lines([('In-sample NAV by version (next-open fill, 1 tick)',
                {e: nav(R['fix'][1][n, 'open']) for e, n in zip(en, names)}, [1])],
              FIG / 'equity_versions.png', 4)
    fig_heatmap(R['grid'], R['params'], FIG / 'heatmap_sharpe.png')
    fig_lines([('Frozen params, in-sample (1 tick)', {FILL_EN[f]: nav(R['cost_is'][1][f, 1][0]) for f in FILLS}, [1]),
               ('Frozen params, out-of-sample (1 tick)', {FILL_EN[f]: nav(R['oos'][1][f, 1][0]) for f in FILLS}, [1])],
              FIG / 'equity_oos.png')


# ---------- 正文 ----------
def write(R):
    figures(R)
    lab, mkt = R['lab'], R['lab'].mkt
    P, sel, st, lv, df = R['params'], R['sel'], R['stat'], R['live'], R['defects']
    o_tab, fx_tab, g = R['orig'][0], R['fix'][0], R['grid']
    c_is, c_oos = R['cost_is'][0], R['oos'][0]
    last = mkt.dates[-1]
    v0c = row(o_tab, scope='full_is', fill='open', ticks=1)
    v0o = row(o_tab, scope='full_is', fill='close', ticks=1)
    v0z = row(o_tab, scope='full_is', fill='open', ticks=0)
    dm = row(o_tab, scope='demo_2409', fill='open', ticks=1)
    names = list(fx_tab['version'].unique())
    fsh = [row(fx_tab, version=n, fill='open')['sharpe'] for n in names]
    far = [row(fx_tab, version=n, fill='open')['ann_ret'] for n in names]
    gi = g[g.fill == 'open']
    pc, po = row(c_is, fill='open', ticks=1), row(c_oos, fill='open', ticks=1)
    pco = row(c_oos, fill='close', ticks=1)
    same = all(sel[k] == P[k] for k in ('window', 'entry', 'exit'))
    yr = R['yearly']
    d_c, d_o = df['open'], df['close']
    oos_t = R['oos'][1]['open', 1][1]
    rolls = R['rolls']

    s = []
    s.append(f"""# MTO 价差策略复盘：复现、缺陷与修正

本报告由 `python run_all.py` 生成，文中数字全部由代码从 `data/` 下两个原始文件算出。样本内 {day(IS_START)} 至 {day(IS_END)}，样本外 {day(OOS_START)} 至 {day(last)}。

## 1 策略概述

来源为天勤量化 tqsdk-python 的示例脚本 `tqsdk/demo/example/mto_spread.py`。甲醇制烯烃（MTO）装置大约 3 吨甲醇产 1 吨烯烃，脚本用 1 手聚乙烯 L（5 吨）加 1 手聚丙烯 PP（5 吨）对 3 手甲醇 MA（每手 10 吨）构造利润价差：

S = 5·L + 5·PP − 30·MA（元/单位）

原脚本逻辑：取当日之前 {W0} 个交易日的 S，算均值 μ 和标准差 σ（np.std，总体口径），z = (S − μ)/σ。空仓时 z > 2 做空利润（卖 L、卖 PP、买 MA），z < −2 做多利润；持仓时 |z| < 0.5 平仓，持空且 z > 3、持多且 z < −3 止损。合约写死为 2409，回测区间写死为 {day(DEMO['start'])} 至 {day(DEMO['end'])}。

本文的步骤：用日线逐条复现原逻辑，给缺陷定量，逐项修正（每步只改一处），只用样本内做参数网格和稳健性检验并冻结参数，最后在样本外跑一次。

## 2 数据与方法

**数据**：tushare `fut_basic`（合约表）与 `fut_daily`（逐合约日线），由 `src/fetch_data.py` 一次性下载，之后只读。文件 sha256：

- `daily_MA_L_PP.csv`：`{R['sha']['daily_MA_L_PP.csv']}`
- `contracts_MA_L_PP.csv`：`{R['sha']['contracts_MA_L_PP.csv']}`

{table(R['summary'].T.rename_axis('项目'), {})}

每手吨数取合约表 `per_unit` 列（MA 10、L 5、PP 5）；tushare 的 `multiplier` 列对商品期货为空。三品种最小变动价位均为 1 元/吨。郑商所 ts_code 用 4 位年月（如 MA2409.ZCE），symbol 用 3 位（MA409）。

**换月**：三腿用同一交割月。每天在候选组里选前一交易日 MA 持仓量最大的那组（MA 主力），只向后换；某组合约在「交割月前一个月的首个交易日」之前 3 个交易日起不再选用，这样收盘成交和次日开盘成交两种口径都能在交割月前一个月之前换完。换月当天平旧开新，两边都计成本。样本内共换月 {rolls['n_is']} 次，换月日距交割月前一个月首日的交易日数中位数为 {rolls['lead_med']:.0f} 天。

**μ、σ 的算法**：每组合约只用它自己过去 w 天的 S。例如换月当天，z 用新组合约过去 w 天的 S 算，不用「旧组历史接新组当日」的拼接序列，避免换月跳空伪造信号。`tests/test_data.py` 对此有专门测试。

**成交与成本**：信号每天收盘算一次。乐观口径按信号日收盘价成交，保守口径按次日开盘价成交（三个品种有夜盘的时期，次日开盘即当晚夜盘开盘）。每腿每边手续费 = 名义额 × {FEE:.4f}，另加滑点 1 跳（另测 0 跳、2 跳）。1 单位 = 1 手 L + 1 手 PP + 3 手 MA，按样本内在持合约的收盘价，1 单位单边成本中位数约 {lv['cost_side_med']:.0f} 元，其中滑点 {lv['slip_side']:.0f} 元。

**收益口径**：日收益率 = 当日净盈亏 / ({MARGIN:.0%} × 前一交易日在持合约 1 单位名义总额)。1 单位名义总额中位数约 {lv['notional_med']/1e4:.1f} 万元，对应保证金约 {lv['margin_med']/1e4:.2f} 万元。净值按日收益简单累加（仓位固定 1 单位，不复利）；年化收益 = 日均收益 × 252；夏普 = 年化收益 / 年化波动（无风险利率取 0，空仓日收益记 0）；最大回撤按累加净值相对前高计算。

**与原脚本的差别**：原脚本每次 K 线变动（盘中每个 tick）都重算 z 并下单，这里只有日线，每天收盘判断一次。原脚本一次下 100 手 L、100 手 PP、300 手 MA，这里按 1 单位算收益率，结果与手数无关（未计冲击成本）。

![价差与 z](fig/spread_z.png)

## 3 原版结果

两种范围：`demo_2409` 为原脚本写死的 2409 合约和回测区间；`full_is` 把同一逻辑放到整个样本内——每组合约一直用到最后交易日（到期收盘强平），次日换到下一组，组的先后顺序与第 2 节换月日历相同，相当于每次合约到期后手工把脚本里的合约代码改成下一个。

{mtable(o_tab, ['scope', 'fill', 'ticks'], '范围 / 成交 / 滑点跳数')}

（fill：close 为乐观口径，open 为保守口径。）交易明细见 `report/trades_original.csv`。

- 原版在样本内保守口径、1 跳成本下年化收益 {pct(v0c.ann_ret)}，夏普 {num(v0c.sharpe)}，最大回撤 {pct(v0c.mdd)}，共 {int(v0c.n)} 笔交易，胜率 {pct(v0c.win)}。乐观口径夏普 {num(v0o.sharpe)}；零成本保守口径夏普 {num(v0z.sharpe)}。
- 在原脚本自己的 2409 合约与区间上（保守口径、1 跳），共 {int(dm.n)} 笔交易，年化收益 {pct(dm.ann_ret)}，夏普 {num(dm.sharpe)}。区间只有半年，交易笔数少，这组数只用来核对复现本身。

![原版净值](fig/equity_original.png)

## 4 缺陷清单

以下定量都取原版样本内结果（1 跳成本），括号前为保守口径，括号内为乐观口径。

1. **止损后没有冷静期**。止损把 `in_position` 置为 False 后，下一次判断时 z 往往仍在 ±2 以外，会按原方向立刻重开。样本内止损 {d_c['n_stop']} 次（{d_o['n_stop']} 次），其中止损次日同方向重开 {d_c['n_reentry']} 次（{d_o['n_reentry']} 次），这些重开交易净盈亏合计 {d_c['reentry_net']:,.0f} 元/单位（{d_o['reentry_net']:,.0f}）。原脚本在盘中每个 tick 都判断，重开可能发生在止损后的下一个 tick，日线回测看不到这一层。
2. **不换月**。合约写死，到期前一个多月流动性已转到下一个合约，交割月前一个月交易所上调保证金、自然人不能进入交割月。样本内有仓位的 {d_c['held_days']} 天中，有 {d_c['late_days']} 天落在交割月前一个月或交割月（其中交割月 {d_c['deliv_days']} 天），这些天的净盈亏合计 {d_c['late_net']:,.0f} 元/单位；这些天在持 MA 合约日成交量中位数 {d_c['vol_late']:,.0f} 手，其余持仓日为 {d_c['vol_normal']:,.0f} 手。到期强平 {d_c['n_expiry']} 次。
3. **平仓条件不分方向**。平仓要求 |z| < 0.5。如果做空后 z 一天内从 2 以上跌到 −0.5 以下，价差已越过均值，脚本仍持仓，要等 z 回到 ±0.5 以内才平，而持空的止损只看 z > 3。样本内有 {d_c['n_overshoot']} 笔交易（{d_o['n_overshoot']} 笔）在持仓期间出现过 z 越过均值到另一侧 0.5 以外仍未平仓。另外平仓阈值取 0 时 |z| < 0 永远不成立，网格里测平仓阈值 0 需要改成方向性判断。
4. **窗口只有 {W0} 天**。样本内去掉换月跳空后的价差 AR(1) 半衰期为 {st['adj_hl']:.1f} 个交易日；每组合约在持期内单独估计，半衰期中位数 {st['per']['hl'].median():.1f} 天。窗口长度与价差回归速度没有对应关系。
5. **`position_time` 记录了没用**，没有持仓时间上限。
6. 其他（日线回测无法定量，只列出）：三腿 K 线按序号对齐而非按日期，某腿缺一根 K 线时三腿会错位；`current_l_pos` 取的是成交回报后的实际持仓，委托未成交时止损判断会失效；回测未计手续费和滑点；100 手 L、300 手 MA 的下单量未考虑盘口深度。

## 5 修正版（样本内，1 跳成本）

每一步只在上一步基础上改一处：

- v1 = v0 + ① 止损冷静期：止损后 |z| 回到 2 以内才允许再开仓
- v2 = v1 + ② 换月：用第 2 节的换月日历，换月时同方向移仓
- v3 = v2 + ③ 窗口：窗口改为样本内半衰期取整 = {R['w3']} 天（规则先定，再跑回测）
- v4 = v3 + ④ 方向性平仓：持空 z < 0.5、持多 z > −0.5 即平（原为 |z| < 0.5）

{mtable(fx_tab, ['version', 'fill'], '版本 / 成交')}

保守口径下各步夏普：""" + '；'.join(f'{n} {num(v)}' for n, v in zip(names, fsh)) + f"""。年化收益：""" + '；'.join(f'{n} {pct(v)}' for n, v in zip(names, far)) + f"""。

![各版本净值](fig/equity_versions.png)

## 6 稳健性（只用样本内）

**参数网格**：在 v4 的规则上（冷静期、换月、方向性平仓），窗口 {{{', '.join(map(str, WINDOWS))}}} × 开仓 {{{', '.join(f'{e:g}' for e in ENTRIES)}}} × 平仓 {{{', '.join(f'{x:g}' for x in EXITS)}}}，止损 = 1.5 × 开仓阈值（沿用原脚本 `STD_THRESHOLD * 1.5` 的写法），1 跳成本。窗口 30 指过去 30 个交易日。

![夏普热力图](fig/heatmap_sharpe.png)

保守口径 36 格中夏普为正的有 {int((gi['sharpe'] > 0).sum())} 格，中位数 {num(gi['sharpe'].median())}，最高 {num(gi['sharpe'].max())}，最低 {num(gi['sharpe'].min())}。

**选参规则**：每格与它在窗口、开仓、平仓三个方向上的相邻格（含对角，边界截断）取保守口径夏普的平均，选平均最高的格，不看单点最高。选出窗口 {P['window']}、开仓 {P['entry']:g}、平仓 {P['exit']:g}：本格夏普 {num(P['sharpe'])}，邻域平均 {num(P['neighborhood_sharpe'])}。单点最高的格是窗口 {P['best_single']['window']}、开仓 {P['best_single']['entry']:g}、平仓 {P['best_single']['exit']:g}，夏普 {num(P['best_single']['sharpe'])}，其邻域平均 {num(P['best_single']['neighborhood_sharpe'])}。参数已写入 `report/params.json` 并单独提交，之后才运行样本外。本次运行按同一规则重算的选择与冻结参数{'一致' if same else '不一致（样本外仍用冻结参数）'}。

**分年度表现**（冻结参数，保守口径，1 跳）：

{table(yr.rename(columns={'ret': '年度收益', 'sharpe': '夏普', 'mdd': '年内最大回撤', 'n': '开仓次数'}).rename_axis('年份'), {'年度收益': pct, '年内最大回撤': pct, '开仓次数': lambda v: f'{int(v)}'})}

样本内 {len(yr)} 个年度中收益为正的有 {int((yr['ret'] > 0).sum())} 个。

**成本敏感性**（冻结参数，样本内）：

{mtable(c_is, ['fill', 'ticks'], '成交 / 滑点跳数')}

**平稳性与半衰期**（样本内）：

| 序列 | ADF 统计量 | p 值 | 滞后阶数 | 样本数 | 半衰期（交易日） |
|---|---|---|---|---|---|
| 去换月跳空的连续价差 | {st['adj_stat']:.2f} | {st['adj_p']:.3f} | {st['adj_lags']} | {st['adj_n']} | {st['adj_hl']:.1f} |

每组合约在持期内单独检验（在持不少于 40 天的 {len(st['per'])} 组）：ADF p < 0.05 的有 {int((st['per']['p'] < 0.05).sum())} 组，p 值中位数 {st['per']['p'].median():.3f}；半衰期中位数 {st['per']['hl'].median():.1f} 天。

## 7 样本外（{day(OOS_START)} 至 {day(last)}，冻结参数，只跑一次）

参数：窗口 {P['window']}、开仓 {P['entry']:g}、平仓 {P['exit']:g}、止损 {P['entry'] * P['stop_mult']:g}，冷静期、换月、方向性平仓均开启。样本外从空仓开始。

{mtable(c_oos, ['fill', 'ticks'], '成交 / 滑点跳数')}

保守口径、1 跳：样本外年化收益 {pct(po.ann_ret)}，夏普 {num(po.sharpe)}，最大回撤 {pct(po.mdd)}，{int(po.n)} 笔交易；同参数样本内夏普 {num(pc.sharpe)}。乐观口径样本外夏普 {num(pco.sharpe)}。

样本外交易（保守口径、1 跳）：

{table(oos_t.assign(side=oos_t.side.map({1: '做多利润', -1: '做空利润'})).set_index('open_exec').rename_axis('开仓成交日')[['side', 'close_exec', 'reason', 'z_open', 'z_close', 'hold_days', 'rolls', 'net']].rename(columns={'side': '方向', 'close_exec': '平仓成交日', 'reason': '平仓原因', 'z_open': '开仓z', 'z_close': '平仓z', 'hold_days': '持仓天数', 'rolls': '换月', 'net': '净盈亏(元/单位)'}), {'开仓z': num, '平仓z': num, '净盈亏(元/单位)': lambda v: f'{v:,.0f}'}) if len(oos_t) else '（无交易）'}

![样本外净值](fig/equity_oos.png)

## 8 实盘视角

**三腿同步成交**：MA 在郑商所，L、PP 在大商所，三腿无法用一张组合单成交，只能分腿下单，先成交的腿要承担其余腿的价格变动。日线数据量化不了盘中分腿的风险，只能给出两个参照：样本内在持组合约的价差从前一日收盘到次日开盘的跳动绝对值中位数 {lv['gap_med']:,.0f} 元/单位、90% 分位 {lv['gap_p90']:,.0f} 元，同期 {W0} 日滚动 σ 中位数 {lv['sd_med']:,.0f} 元；冻结参数下保守口径与乐观口径的样本内夏普分别为 {num(row(c_is, fill='open', ticks=1).sharpe)} 和 {num(row(c_is, fill='close', ticks=1).sharpe)}，差值反映成交时点晚半天的代价。

**保证金**：按 {MARGIN:.0%} 计，1 单位保证金中位数约 {lv['margin_med']/1e4:.2f} 万元。冻结参数样本内（保守口径）单笔交易持仓期间最大浮亏中位数为保证金的 {pct(lv['mae_med'])}，最差一笔为 {pct(lv['mae_worst'])}。实盘还要留出交易所临近交割月上调保证金和期货公司加收的部分；换月规则让持仓不进入交割月前一个月，避开了临近交割的保证金上调。

**流动性**：样本内在持合约日成交量中位数 L {lv['vol_med']['L']:,.0f} 手、PP {lv['vol_med']['PP']:,.0f} 手、MA {lv['vol_med']['MA']:,.0f} 手；换月当天新组合约的成交量中位数 L {lv['vol_roll']['L']:,.0f}、PP {lv['vol_roll']['PP']:,.0f}、MA {lv['vol_roll']['MA']:,.0f} 手。原脚本的下单量（100 手 L、100 手 PP、300 手 MA）分别占上述日成交量中位数的 {pct(lv['demo_share']['L'], 2)}、{pct(lv['demo_share']['PP'], 2)}、{pct(lv['demo_share']['MA'], 2)}。日成交量只说明全天容量，开盘和夜盘开盘时点的盘口深度需要逐笔数据另行评估。

## 9 结论与局限

""")
    s.append(conclusions(R, v0c, fsh, far, gi, P, pc, po, same))
    s.append(f"""
**局限**：
- 只有日线，信号每天判断一次，盘中触发、分腿成交、冲击成本都没有模拟；滑点按固定跳数假设，没有盘口数据校准。
- 手续费按名义额万分之一估计，与各期货公司实际费率不同；成本敏感性表给出 0、1、2 跳的结果。
- 10 吨/手的 MA 合约 2014 年 6 月才上市，样本内从 {day(IS_START)} 开始，前几个月在持的 MA 合约还不是全市场主力（当时主力是 50 吨/手的老合约，不在数据里）。
- 样本外约 {R['oos_years']:.1f} 年，交易 {int(po.n)} 笔，统计意义有限。
- 选参只看夏普一个指标，邻域取平均的规则本身也是一种选择。

## 附：复现

```bash
pip install -r requirements.txt
python run_all.py        # 重算全部表格、图和本报告
pytest -q                # 数据层与回测引擎测试
```

文件：`src/data.py` 数据层与换月日历；`src/backtest.py` 回测引擎；`src/analysis.py` 各任务的计算；`src/report.py` 图与报告；`report/params.json` 冻结参数；`report/trades_original.csv` 原版交易明细；`report/trades_oos.csv` 样本外交易；`report/roll_calendar.csv` 换月记录。
""")
    (REPORT / 'report.md').write_text('\n'.join(s))


def conclusions(R, v0c, fsh, far, gi, P, pc, po, same):
    out = []
    out.append(f"1. 原版逻辑在样本内扣除 1 跳成本后（保守口径）年化收益 {pct(v0c.ann_ret)}、夏普 {num(v0c.sharpe)}，"
               f"{'不具备可交易的收益' if v0c.sharpe < 0.5 else '有一定收益'}。")
    out.append(f"2. 四处修正累加后（v4，保守口径）夏普由 {num(fsh[0])} {up_down(fsh[0], fsh[-1])} {num(fsh[-1])}，"
               f"年化收益由 {pct(far[0])} 变为 {pct(far[-1])}。各步贡献见第 5 节表格。")
    out.append(f"3. 参数网格保守口径 36 格中 {int((gi['sharpe'] > 0).sum())} 格夏普为正；按邻域平均选出的参数样本内夏普 {num(pc.sharpe)}。")
    out.append(f"4. 冻结参数在样本外（保守口径、1 跳）夏普 {num(po.sharpe)}、年化收益 {pct(po.ann_ret)}、{int(po.n)} 笔交易，"
               f"{'方向与样本内一致' if np.sign(po.ann_ret) == np.sign(pc.ann_ret) else '与样本内方向相反'}。")
    return '\n'.join(out) + '\n'
