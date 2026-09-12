"""生成 report/fig/*.png 与 report/report.md。数字全部取自 run_all.py 算出的结果，不手填。

正文按第二轮写；第一轮（原设计，唯一一次干净的样本外）的结果与两轮差别在第 1 节。
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import (CAPITAL, DEMO, ENTRIES, EXITS, FILLS, MAIN_MONTHS, MONEY_STOP, REPORT, SIZE_CAP,  # noqa: E402
                       VARIANTS, VOL_WIN, W0, WF_TRAIN, WINDOWS, calendar_diff, sharpe_stats)
from .backtest import FEE, MARGIN  # noqa: E402
from .data import L_TICK5_BEFORE, pick  # noqa: E402

FIG = REPORT / 'fig'
FILL_EN = {'close': 'close fill', 'open': 'next-open fill'}
FILL_CN = {'close': '乐观', 'open': '保守'}
COLS = {'ann_ret': '年化收益', 'mdd': '最大回撤', 'sharpe': '夏普', 'win': '胜率', 'n': '交易次数',
        'hold': '平均持仓天数', 'n_stop': '止损次数', 'n_roll': '换月次数'}
# 样本内起点的依据（事实与出处，非计算结果）
TIMELINE = [
    ('2014-12-12', '郑商所开设夜盘，甲醇在首批品种里，夜盘 21:00–23:30',
     '[郑商所官网转载的上海证券报报道](https://www.czce.com.cn/cn/ypzt/mtbd/webinfo/2014/12/1415698823450310.htm)'),
    ('2019-03-29', '大商所给聚乙烯、聚丙烯等增加夜盘，21:00–23:00（此前这两个品种没有夜盘）',
     '[大商所发〔2019〕124 号](http://www.dce.com.cn/dce/content/2019/ywggytz/6156940.html)'),
    ('2019-12-12', '郑商所自 2019-12-11 晚起夜盘统一到 23:00 结束；从 12-12 这个交易日起三腿交易时段完全一致',
     '郑商所 2019-12-10 通知（未找到官网原文，据[搜狐](https://www.sohu.com/a/359593738_555060)等媒体转载）'),
    ('2021-11-01', '大商所聚乙烯最小变动价位由 5 元/吨调为 1 元/吨（10-29 夜盘起）',
     '[大商所公告](http://www.dce.com.cn/dalianshangpin/ywfw/jystz/ywtz/6293982/index.html)'),
    ('2022-08-01', '《中华人民共和国期货和衍生品法》施行',
     '[中国人大网](http://www.npc.gov.cn/npc/c2/c30834/202204/t20220420_317569.html)'),
    ('2025-10-09', '《期货市场程序化交易管理规定（试行）》施行（在样本外期间）',
     '[证监会公告〔2025〕12 号](https://www.csrc.gov.cn/csrc/c101954/c7564346/content.shtml)'),
]


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
    for idx, r in df.iterrows():
        out.append('| ' + ' | '.join([str(idx)] + [fmt.get(c, str)(v) for c, v in r.items()]) + ' |')
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


# ---------- 图（第二轮）----------
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
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
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
            if r == 1:
                ax.set_xlabel('entry |z|')
            if c == 0:
                ax.set_ylabel('window (days)')
            ax.set_title(f'{FILL_EN[fill]}, exit={x:g}', fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.6, label='Sharpe (in-sample, 1 tick)')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def figures(R, R1):
    FIG.mkdir(parents=True, exist_ok=True)
    lab, stat, r = R['lab'], R['stat'], R['r']
    cal = lab.cal['roll']
    Sa = pick(lab.mkt.S_close, cal).loc[r.is_start:]
    za = pick(lab.z(W0), cal).loc[r.is_start:]
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
    fig_lines([('Round 2 frozen params, in-sample (1 tick)',
                {FILL_EN[f]: nav(R['cost_is'][1][f, 1][0]) for f in FILLS}, [1]),
               ('Out-of-sample (1 tick): round 2 (period already seen) vs round 1 (the clean test)',
                {**{f'round 2, {FILL_EN[f]}': nav(R['oos'][1][f, 1][0]) for f in FILLS},
                 **{f'round 1, {FILL_EN[f]}': nav(R1['oos'][1][f, 1][0]) for f in FILLS}}, [1])],
              FIG / 'equity_oos.png')


def fig_walkforward(R3, path):
    nav3 = {v: 1 + R3['runs'][v]['open'][0]['ret'].cumsum() for v in VARIANTS}
    en = {VARIANTS[0]: 'A fixed size', VARIANTS[1]: 'B vol-scaled size', VARIANTS[2]: 'C + money stop'}
    fig_lines([(f'Walk-forward NAV (next-open fill, 1 tick, capital = {CAPITAL}x margin); '
                'each year uses only parameters chosen on the 3 preceding years',
                {en[v]: s for v, s in nav3.items()}, [1])], path, 4)


def round3_section(R3):
    folds, m = R3['folds'], R3['metrics']
    agg = m[m.fill == 'open'].set_index(['variant', 'scope'])
    years = sorted(folds.year.unique())
    rows = []
    for v in VARIANTS:
        for scope, label in (('all', f'{years[0]}–{years[-1]} 全部'), ('since2020', '2020 起')):
            x = agg.loc[(v, scope)]
            rows.append({'风控': v, '检验年': label, '年化收益': x.ann_ret, '最大回撤': x.mdd, '夏普': x.sharpe,
                         '夏普标准误': x.se, 't 值': x.t, '交易次数': x.n, '胜率': x.win})
    t_agg = pd.DataFrame(rows).set_index('风控')
    fr = []
    for y in years:
        f0 = folds[folds.year == y]
        d = {'训练区间': f0.train.iloc[0]}
        for v in VARIANTS:
            x = f0[f0.variant == v].iloc[0]
            d[f'{v[0]} 参数'] = f'{x.window}/{x.entry:g}/{x.exit:g}'
            d[f'{v[0]} 收益'] = x.ann_ret
        fr.append(pd.Series(d, name=y))
    t_fold = pd.DataFrame(fr).rename_axis('检验年')
    fmt = {**{f'{v[0]} 收益': pct for v in VARIANTS}, '夏普标准误': num, 't 值': num}
    a, b, c = (agg.loc[(v, 'all')] for v in VARIANTS)
    nb = agg.loc[(VARIANTS[1], 'since2020')]
    zero = [v for v, x in zip(VARIANTS, (a, b, c)) if abs(x.sharpe) < x.se]
    nwin = folds.groupby('variant')['window'].nunique().max()
    cw = R3['cost'].pivot(index='ticks', columns='variant', values='sharpe')
    t_cost = cw.rename_axis('滑点跳数').rename(columns={v: v.split(' ')[1] for v in VARIANTS})
    lo, hi = R3['cost'].sharpe.min(), R3['cost'].sharpe.max()
    se_max = R3['cost'].se.max()
    cost_fmt = {c: num for c in t_cost.columns}
    return f"""
## 11 滚动检验与风控（第三轮）

前两轮的样本外都只有 1.2 年、7 笔交易，说明不了问题，而 {day(R3['lab'].mkt.dates[-1])} 之前的数据已经全部看过，新规则没有干净的样本外可用。第三轮改用滚动检验：检验年 {years[0]}–{years[-1]}，每个检验年只用它之前 {WF_TRAIN} 个自然年的数据、按与前两轮相同的网格和邻域规则选参数，再只跑这一年。每个检验年对它自己那组参数都是没看过的数据。检验年末强平，跨年不留仓。

信号规则沿用第二轮 v4（冷静期、只在 {'、'.join(MAIN_MONTHS)} 换月、方向性平仓、止损 1.5 × 开仓阈值），在它上面逐档加风控：

- A 固定 1 单位
- B 按波动定手数：手数 = 训练窗口内 σ 的中位数 / 当前 σ，σ 为在持组过去 {VOL_WIN} 个交易日 S 日变化的标准差（只用当日之前），手数截断到 [{SIZE_CAP[0]:g}, {SIZE_CAP[1]:g}]，开仓时定死
- C 在 B 上加金额止损：持仓浮亏达到该仓位保证金的 {MONEY_STOP:.0%} 即平

本金口径改为 {CAPITAL} 倍 1 单位保证金（{1 / (CAPITAL * MARGIN):.1f} 倍杠杆）。夏普不随本金变，年化收益与回撤同时除以 {CAPITAL}。下表为保守口径、1 跳成本；2019 整年在三腿交易时段一致之前，所以另列 2020 起的合计。

{table(t_agg, fmt)}

各检验年选出的参数（窗口/开仓/平仓）与当年收益：

{table(t_fold, fmt)}

![滚动检验净值](fig/walkforward.png)

几点：

- 三档风控的夏普{'都' if len(zero) == len(VARIANTS) else ''}在一个标准误之内包含 0（{'、'.join(f'{v[0]} {num(agg.loc[(v, "all")].sharpe)}±{num(agg.loc[(v, "all")].se)}' for v in VARIANTS)}），按第三轮事先写下的判定标准，这套规则在滚动检验下没有可验证的边际。
- 按波动定手数（B 对 A）同时改善了两头：夏普 {num(a.sharpe)}→{num(b.sharpe)}，最大回撤 {pct(a.mdd)}→{pct(b.mdd)}。把本金放到 {CAPITAL} 倍保证金之后，回撤已经落到 {pct(b.mdd)}，说明前面几轮 90% 的回撤主要来自本金口径（1 倍保证金、{1 / MARGIN:.1f} 倍杠杆），而不是交易本身。
- 金额止损（C 对 B）两头都变差：夏普 {num(b.sharpe)}→{num(c.sharpe)}，最大回撤 {pct(b.mdd)}→{pct(c.mdd)}，交易次数 {int(b.n)}→{int(c.n)} 笔。它砍掉的是后来会回归的仓位，同时多付了成本，与第 5 节第 1 条、第 6 节冷静期那一步是同一件事的两面。
- 各检验年选出的窗口在 {nwin} 个取值之间跳（见上表），样本内最优参数本身不稳定。

**成本敏感性**：每个滑点档都按该档的成本假设重新选参数，合计夏普（保守口径）：

{table(t_cost, cost_fmt)}

零成本下三档都是负的；滑点提高反而有的变好，因为成本假设一变，每年选出的参数就跟着变。三个档位之间的差别（{num(lo)} 到 {num(hi)}）小于夏普的标准误（约 {num(se_max)}），属于噪声。结论不是被成本压掉的。
"""


# ---------- 正文 ----------
def same_oos(R1, R2):
    """两轮样本外（所有成交口径与滑点档）的交易明细和逐日收益是否完全相同。"""
    return all(R1['oos'][1][k][1].equals(R2['oos'][1][k][1]) and R1['oos'][1][k][0]['ret'].equals(R2['oos'][1][k][0]['ret'])
               for k in R2['oos'][1])


def round1_section(R1, R2):
    o, f, ci, co, P1 = R1['orig'][0], R1['fix'][0], R1['cost_is'][0], R1['oos'][0], R1['params']
    v4 = f['version'].iloc[-1]
    rows = []
    for label, df, kw in (('原版（样本内）', o, dict(scope='full_is', ticks=1)), (f'修正版 {v4[:2]}（样本内）', f, dict(version=v4)),
                          ('冻结参数（样本内）', ci, dict(ticks=1)), ('冻结参数（样本外）', co, dict(ticks=1))):
        for fill in ('open', 'close'):
            x = row(df, fill=fill, **kw)
            rows.append({'结果': label, '口径': FILL_CN[fill], '年化收益': x.ann_ret, '最大回撤': x.mdd,
                         '夏普': x.sharpe, '交易次数': x.n})
    t1 = pd.DataFrame(rows).set_index('结果')
    r1, r2 = R1['r'], R2['r']
    lag = R1['rolls']['lag']
    p2 = R2['params']
    same_p = all(P1[k] == p2[k] for k in ('window', 'entry', 'exit'))
    same = same_oos(R1, R2)
    cd = calendar_diff(R1, R2)
    if cd['n']:
        cal_txt = (f"样本外期间两轮的换月日历有 {cd['n']} 个交易日不同（{day(cd['first'])} 至 {day(cd['last'])}，第一轮在持 "
                   f"{'、'.join(cd['m1'])} 组，第二轮在持 {'、'.join(cd['m2'])} 组），这些天两轮{'都空仓' if not cd['held'] else '有仓位'}")
    else:
        cal_txt = '样本外期间两轮的换月日历完全相同'
    after = ((f"第二轮按同一选参规则在新样本内选出的参数与第一轮相同（窗口 {p2['window']}、开仓 {p2['entry']:g}、平仓 {p2['exit']:g}）。"
              if same_p else f"第二轮选出的参数为窗口 {p2['window']}、开仓 {p2['entry']:g}、平仓 {p2['exit']:g}。")
             + f"{cal_txt}。" + ("所以第二轮样本外交易与第一轮逐笔相同，第二轮的修改没有改变样本外结果。" if same else ''))
    return f"""## 1 两轮说明与第一轮结果

第一轮按任务书的顺序完成：只用样本内（{day(r1.is_start)} 至 {day(r1.is_end)}）选参，单独提交冻结参数 `report/round1/params.json`，再在样本外只跑一次。这是本项目唯一一次干净的样本外检验。第一轮冻结参数为窗口 {P1['window']}、开仓 {P1['entry']:g}、平仓 {P1['exit']:g}，结果（1 跳成本；保守＝次日开盘成交，乐观＝信号日收盘成交）：

{table(t1)}

`run_all.py` 每次都重算第一轮，并核对冻结参数与样本外结果文件和首次运行时入库的版本逐字节一致。

看过第一轮样本外之后，按要求改了三处，形成第二轮：

| 项目 | 第一轮 | 第二轮 | 原因 |
|---|---|---|---|
| 样本内起点 | {day(r1.is_start)}（按流动性定） | {day(r2.is_start)} | 三腿交易时段从这个交易日起完全一致，见第 3 节时间线 |
| 换月候选 | 全部交割月 | 只在 {'、'.join(MAIN_MONTHS)} 月 | 第一轮样本内有 {R1['rolls']['n_odd']} 次先换到非 01/05/09 月份，{lag[0]}–{lag[1]} 个交易日内又换一次，有持仓时多付一次换月成本 |
| ③ 窗口 | 不超过半衰期的最大网格窗口，{R1['w3']} 天 | 半衰期取整；在持组自有历史不够时取样本内每天都算得出 z 的最长窗口，{R2['w3']} 天 | 第一轮这条规则是看过样本内网格之后改的 |

成本、成交口径、网格、选参规则、止损倍数、冷静期、方向性平仓都没变。第二轮的设计在跑之前写进 PROGRESS.md，冻结参数 `report/round2/params.json` 在样本内选定后单独提交，之后才首次运行第二轮样本外。但这三处修改是在看过第一轮样本外之后做的，{day(r2.oos_start)} 以后的数据对第二轮已经不是独立检验，第二轮样本外只作参考；要检验第二轮，需要 {day(R2['lab'].mkt.dates[-1])} 以后的新数据。

{after}

以下第 2–10 节为第二轮，第 11 节为第三轮。
"""


def write(R, R1, R3, meta):
    figures(R, R1)
    fig_walkforward(R3, FIG / 'walkforward.png')
    lab, mkt, r = R['lab'], R['lab'].mkt, R['r']
    P, sel, st, lv, df = R['params'], R['sel'], R['stat'], R['live'], R['defects']
    o_tab, fx_tab, g = R['orig'][0], R['fix'][0], R['grid']
    c_is, c_oos = R['cost_is'][0], R['oos'][0]
    last = mkt.dates[-1]
    v0c = row(o_tab, scope='full_is', fill='open', ticks=1)
    v0o = row(o_tab, scope='full_is', fill='close', ticks=1)
    v0z = row(o_tab, scope='full_is', fill='open', ticks=0)
    dm = row(o_tab, scope='demo_2409', fill='open', ticks=1)
    names = list(fx_tab['version'].unique())
    fx = {n: row(fx_tab, version=n, fill='open') for n in names}
    steps = [f'{names[k]}：夏普 {num(fx[names[k - 1]].sharpe)}→{num(fx[names[k]].sharpe)}，最大回撤 '
             f'{pct(fx[names[k - 1]].mdd)}→{pct(fx[names[k]].mdd)}，交易 {int(fx[names[k - 1]].n)}→{int(fx[names[k]].n)} 笔'
             for k in range(1, len(names))]
    gi = g[g.fill == 'open']
    pc, pco_is = row(c_is, fill='open', ticks=1), row(c_is, fill='close', ticks=1)
    po, pco = row(c_oos, fill='open', ticks=1), row(c_oos, fill='close', ticks=1)
    q1, q1c = row(R1['oos'][0], fill='open', ticks=1), row(R1['oos'][0], fill='close', ticks=1)
    same = all(sel[k] == P[k] for k in ('window', 'entry', 'exit'))
    edge = [t for t, ok in (('窗口取最大值', P['window'] == max(WINDOWS)), ('窗口取最小值', P['window'] == min(WINDOWS)),
                            ('开仓取最大值', P['entry'] == max(ENTRIES)), ('开仓取最小值', P['entry'] == min(ENTRIES)),
                            ('平仓取最大值', P['exit'] == max(EXITS)), ('平仓取最小值', P['exit'] == min(EXITS))) if ok]
    yr = R['yearly']
    d_c, d_o = df['open'], df['close']
    rolls, ts = R['rolls'], R['ts']
    ti = R['cost_is'][1]['open', 1][1]
    side = ti.groupby('side')['net'].agg(['size', 'sum']).reindex([-1, 1]).fillna(0)
    oos_t = R['oos'][1]['open', 1][1]
    mg = R['oos'][1]['close', 1][1].merge(oos_t, on='open_signal', suffixes=('_c', '_o'))
    gap_txt, one_jump = '', False
    if len(mg):
        gd = mg['pnl_c'] - mg['pnl_o']
        k = gd.abs().idxmax()
        one_jump = abs(gd[k]) >= 0.8 * abs(gd.sum())
        kd, km = mg.at[k, 'open_signal'], mg.at[k, 'month_c']
        s0, s1 = mkt.S_close.at[kd, km], mkt.S_open.iat[mkt.dates.searchsorted(kd) + 1, mkt.months.index(km)]
        gap_txt = (f"两种口径的信号完全相同，差别只在成交价。样本外 {len(mg)} 笔交易的毛盈亏，乐观口径合计 {mg['pnl_c'].sum():,.0f} 元/单位，"
                   f"保守口径 {mg['pnl_o'].sum():,.0f}，差 {gd.sum():,.0f}；其中差距最大的是 {day(kd)} 发出信号的那笔（{gd[k]:,.0f}），"
                   f"当晚在持组 S 从收盘 {s0:,.0f} 元到次日开盘 {s1:,.0f} 元。")
    if R['w3'] < (R['w_hl'] or 10 ** 6):
        v3_txt = (f"样本内连续价差半衰期取整为 {R['w_hl']} 天，但在持组自有历史最短只有 {R['w_hist']} 天"
                  f"（{R['w_hl']} 天窗口下样本内 {R['cov_hl'][1]} 个交易日里 z 能算出 {R['cov_hl'][0]} 天），所以取 {R['w3']} 天")
    else:
        v3_txt = f"样本内连续价差半衰期取整为 {R['w_hl']} 天，在持组自有历史最短 {R['w_hist']} 天，够用，取 {R['w3']} 天"
    ev = meta['ticks']
    tl = '\n'.join(f'| {d} | {e} | {s} |' for d, e, s in TIMELINE)
    same_o = same_oos(R1, R)
    reent = ('在这段样本里，止损出场后按原方向重开整体是赚钱的，说明止损之后价差多数又回归了；第 6 节加冷静期后夏普下降，主要来自去掉了这些交易。'
             if d_c['reentry_net'] > 0 else '')
    same_v34 = fx[names[-1]][list(COLS)].equals(fx[names[-2]][list(COLS)])
    v34_txt = (f"{names[-1]} 与 {names[-2]} 结果完全相同：两种平仓规则只在持仓期间 z 越过均值到另一侧 0.5 以外时才有差别，"
               "样本内没有出现这种情况。" if same_v34 else '')
    oos_cmp = ('两轮样本外交易逐笔相同（原因见第 1 节），这组数也就是第一轮那次干净检验的结果。' if same_o else
               f"同一段样本外，第一轮（干净的检验）保守口径夏普 {num(q1.sharpe)}、年化 {pct(q1.ann_ret)}，乐观口径夏普 {num(q1c.sharpe)}。")
    oos_tab = (oos_t.assign(side=oos_t.side.map({1: '做多利润', -1: '做空利润'}), open_exec=oos_t.open_exec.map(day),
                            close_exec=oos_t.close_exec.map(day)).set_index('open_exec').rename_axis('开仓成交日')
               [['side', 'close_exec', 'reason', 'z_open', 'z_close', 'hold_days', 'rolls', 'net']]
               .rename(columns={'side': '方向', 'close_exec': '平仓成交日', 'reason': '平仓原因', 'z_open': '开仓z',
                                'z_close': '平仓z', 'hold_days': '持仓天数', 'rolls': '换月', 'net': '净盈亏(元/单位)'}))

    s = [f"""# MTO 价差策略复盘：复现、缺陷与修正

本报告由 `python run_all.py` 生成，文中数字全部由代码从 `data/` 下两个原始文件算出。项目做了三轮：第一轮是原始设计（样本内 {day(R1['r'].is_start)} 起，样本外 {day(r.oos_start)} 至 {day(last)}，唯一一次干净的样本外检验）；第二轮是看过第一轮样本外之后按要求做的修订（样本内 {day(r.is_start)} 至 {day(r.is_end)}）；第三轮改用滚动检验，并加了资金与风控，见第 11 节。

""", round1_section(R1, R), f"""
## 2 策略概述

来源为天勤量化 tqsdk-python 的示例脚本 `tqsdk/demo/example/mto_spread.py`。甲醇制烯烃（MTO）装置大约 3 吨甲醇产 1 吨烯烃，脚本用 1 手聚乙烯 L（5 吨）加 1 手聚丙烯 PP（5 吨）对 3 手甲醇 MA（每手 10 吨）构造利润价差：

S = 5·L + 5·PP − 30·MA（元/单位）

原脚本逻辑：取当日之前 {W0} 个交易日的 S，算均值 μ 和标准差 σ（np.std，总体口径），z = (S − μ)/σ。空仓时 z > 2 做空利润（卖 L、卖 PP、买 MA），z < −2 做多利润；持仓时 |z| < 0.5 平仓，持空且 z > 3、持多且 z < −3 止损。合约写死为 2409，回测区间写死为 {day(DEMO['start'])} 至 {day(DEMO['end'])}。

步骤：用日线逐条复现原逻辑，给缺陷定量，逐项修正（每步只改一处），只用样本内做参数网格和稳健性检验并冻结参数，最后在样本外运行。

## 3 数据与方法

**数据**：tushare `fut_basic`（合约表）与 `fut_daily`（逐合约日线），由 `src/fetch_data.py` 一次性下载，之后只读。文件 sha256：

- `daily_MA_L_PP.csv`：`{meta['sha']['daily_MA_L_PP.csv']}`
- `contracts_MA_L_PP.csv`：`{meta['sha']['contracts_MA_L_PP.csv']}`

{table(meta['summary'].T.rename_axis('项目'), {})}

每手吨数取合约表 `per_unit` 列（MA 10、L 5、PP 5）；tushare 的 `multiplier` 列对商品期货为空。郑商所 ts_code 用 4 位年月（如 MA2409.ZCE），symbol 用 3 位（MA409）。

**最小变动价位**：合约表对三个品种都写 1 元/吨。有成交的行里开高低收全是 5 的整数倍的比例，L 在 {day(L_TICK5_BEFORE)} 之前为 {pct(ev.loc['L', '之前'])}、之后为 {pct(ev.loc['L', '之后'])}，PP、MA 在该日之前分别为 {pct(ev.loc['PP', '之前'])}、{pct(ev.loc['MA', '之前'])}；大商所公告（见下表）证实聚乙烯自 {day(L_TICK5_BEFORE)} 起由 5 元/吨改为 1 元/吨。L 在此之前按 5 元/吨一跳计滑点，之后按 1 元/吨；PP、MA 始终 1 元/吨。

**无成交日**：日线中收盘价缺失 {meta['fills']['收盘用结算价补']:,} 行，用当日结算价补；开盘价缺失 {meta['fills']['开盘用收盘价补']:,} 行，用补后的收盘价补（共 {meta['fills']['总行数']:,} 行）。期货按结算价盯市，这条规则主要影响原版持有到期那几天。

**样本内起点**：按「从最近一次对这三个合约交易有重大影响的政策或规则变化之后开始」的要求，查了以下事件：

| 日期 | 事件 | 出处 |
|---|---|---|
{tl}

2019-03-29 之前甲醇有夜盘而聚乙烯、聚丙烯没有：保守口径用的次日开盘价，甲醇是前一晚 21:00 的价格，另两腿是次日 9:00 的价格，不是同一时刻，隔夜消息先反映在甲醇上。2019-03-29 至 2019-12-11 三腿都从 21:00 开盘，但甲醇夜盘多交易半小时，23:00–23:30 的消息只反映在甲醇上。第二轮样本内从 {day(r.is_start)} 开始，此后三个合约的交易规则只有聚乙烯跳价一处变化（已按时间计入成本）；期货法施行当天没有改这三个合约的交易规则，大商所据此修订交易规则的公告在 2026 年 1 月，已在样本外期间。起点附近的滚动窗口会用到起点之前这组合约自己的价格，只作指标预热，交易、平稳性检验、网格与选参都从起点算。

**换月**：三腿用同一交割月，只在 {'、'.join(MAIN_MONTHS)} 三个月份里选。每天在候选组里选前一交易日 MA 持仓量最大的那组，只向后换；某组合约在「交割月前一个月的首个交易日」之前 3 个交易日起不再选用，收盘成交和次日开盘成交两种口径都能在交割月前一个月之前换完。换月当天平旧开新，两边都计成本。第二轮样本内换月 {rolls['n_is']} 次，换月日距交割月前一个月首日的交易日数中位数为 {rolls['lead_med']:.0f} 天{'，多数换月在截止日触发' if rolls['lead_med'] <= 2 else ''}，换到非 01/05/09 月份 {rolls['n_odd']} 次。

**μ、σ 的算法**：每组合约只用它自己过去 w 天的 S。换月当天的 z 用新组合约过去 w 天的 S 算，不用「旧组历史接新组当日」的拼接序列，避免换月跳空伪造信号。`tests/test_data.py` 对此有专门测试。

**成交与成本**：信号每天收盘算一次。乐观口径按信号日收盘价成交，保守口径按次日开盘价成交（三个品种都有夜盘，次日开盘即当晚夜盘开盘）。每腿每边手续费 = 名义额 × {FEE:.4f}，另加滑点 1 跳（另测 0 跳、2 跳）。1 单位 = 1 手 L + 1 手 PP + 3 手 MA，按样本内在持合约的收盘价，1 单位单边成本中位数约 {lv['cost_side_med']:.0f} 元；其中 1 跳滑点在 {day(L_TICK5_BEFORE)} 前为 {lv['slip_before']:.0f} 元、之后为 {lv['slip_after']:.0f} 元。

**收益口径**：日收益率 = 当日净盈亏 / ({MARGIN:.0%} × 前一交易日在持合约 1 单位名义总额)。1 单位名义总额中位数约 {lv['notional_med']/1e4:.1f} 万元，对应保证金约 {lv['margin_med']/1e4:.2f} 万元。净值按日收益简单累加（仓位固定 1 单位，不复利）；年化收益 = 日均收益 × 252；夏普 = 年化收益 / 年化波动（无风险利率取 0，空仓日收益记 0）；最大回撤按累加净值相对前高计算。

**与原脚本的差别**：原脚本每次 K 线变动（盘中每个 tick）都重算 z 并下单，这里只有日线，每天收盘判断一次。原脚本一次下 100 手 L、100 手 PP、300 手 MA，这里按 1 单位算收益率，结果与手数无关（未计冲击成本）。

![价差与 z](fig/spread_z.png)

## 4 原版结果

两种范围：`demo_2409` 为原脚本写死的 2409 合约和回测区间；`full_is` 把同一逻辑放到第二轮样本内——每组合约一直用到最后交易日（到期收盘强平），次日换到下一组，组的先后顺序与第 3 节换月日历相同，相当于每次合约到期后手工把脚本里的合约代码改成下一个。

{mtable(o_tab, ['scope', 'fill', 'ticks'], '范围 / 成交 / 滑点跳数')}

（fill：close 为乐观口径，open 为保守口径。）交易明细见 `report/round2/trades_original.csv`。

- 原版在样本内保守口径、1 跳成本下年化收益 {pct(v0c.ann_ret)}，夏普 {num(v0c.sharpe)}，最大回撤 {pct(v0c.mdd)}，共 {int(v0c.n)} 笔交易，胜率 {pct(v0c.win)}。乐观口径夏普 {num(v0o.sharpe)}；零成本保守口径夏普 {num(v0z.sharpe)}。{'最大回撤超过 100%，按 12% 保证金持 1 单位，回撤期间亏掉的钱超过全部保证金，实盘会被强平。' if v0c.mdd > 1 else ''}
- 在原脚本自己的 2409 合约与区间上（保守口径、1 跳），共 {int(dm.n)} 笔交易，年化收益 {pct(dm.ann_ret)}，夏普 {num(dm.sharpe)}。区间只有半年；同一逻辑放到整个样本内夏普为 {num(v0c.sharpe)}，{'原脚本自带的回测区间恰好是这个策略表现好的一段。' if dm.sharpe > v0c.sharpe + 0.5 else '两者差别不大。'}

![原版净值](fig/equity_original.png)

## 5 缺陷清单

以下定量都取原版在第二轮样本内的结果（1 跳成本），括号前为保守口径，括号内为乐观口径。

1. **止损后没有冷静期**。止损把 `in_position` 置为 False 后，下一次判断时 z 往往仍在 ±2 以外，会按原方向立刻重开。样本内止损 {d_c['n_stop']} 次（{d_o['n_stop']} 次），其中止损次日同方向重开 {d_c['n_reentry']} 次（{d_o['n_reentry']} 次），这些重开交易净盈亏合计 {d_c['reentry_net']:,.0f} 元/单位（{d_o['reentry_net']:,.0f}），胜率 {pct(d_c['reentry_win'])}（{pct(d_o['reentry_win'])}）。{reent}原脚本在盘中每个 tick 都判断，重开可能发生在止损后的下一个 tick，日线回测看不到这一层。
2. **不换月**。合约写死，到期前一个多月流动性已转到下一个合约，临近交割月交易所分段提高保证金，个人客户的持仓不能进入交割月。样本内有仓位的 {d_c['held_days']} 天中，有 {d_c['late_days']} 天落在交割月前一个月或交割月（其中交割月 {d_c['deliv_days']} 天），这些天的净盈亏合计 {d_c['late_net']:,.0f} 元/单位（{d_o['late_net']:,.0f}），全部持仓日合计 {d_c['total_net']:,.0f}（{d_o['total_net']:,.0f}）；这些天在持 MA 合约日成交量中位数 {d_c['vol_late']:,.0f} 手，其余持仓日为 {d_c['vol_normal']:,.0f} 手。到期强平 {d_c['n_expiry']} 次。
3. **平仓条件不分方向**。平仓要求 |z| < 0.5。如果做空后 z 一天内从 2 以上跌到 −0.5 以下，价差已越过均值，脚本仍持仓，要等 z 回到 ±0.5 以内才平，而持空的止损只看 z > 3。样本内有 {d_c['n_overshoot']} 笔交易（{d_o['n_overshoot']} 笔）在持仓期间出现过 z 越过均值到另一侧 0.5 以外仍未平仓。另外平仓阈值取 0 时 |z| < 0 永远不成立，网格里测平仓阈值 0 需要改成方向性判断。
4. **窗口 {W0} 天与价差回归速度对不上**。样本内去掉换月跳空后的连续价差，AR(1) 半衰期为 {st['adj_hl']:.1f} 个交易日，ADF 检验 p 值 {st['adj_p']:.2f}；每组合约在持期内单独估计，半衰期中位数 {st['per']['hl'].median():.1f} 天，{len(st['per'])} 组里 {int((st['per']['p'] < 0.05).sum())} 组 ADF 在 5% 水平拒绝单位根。
5. **`position_time` 记录了没用**，没有持仓时间上限。
6. 其他（日线回测无法定量，只列出）：三腿 K 线按序号对齐而非按日期，某腿缺一根 K 线时三腿会错位；`current_l_pos` 取的是成交回报后的实际持仓，委托未成交时止损判断会失效；100 手 L、300 手 MA 的下单量未考虑盘口深度。

## 6 修正版（第二轮样本内，1 跳成本）

每一步只在上一步基础上改一处：

- v1 = v0 + ① 止损冷静期：止损后 |z| 回到 2 以内才允许再开仓
- v2 = v1 + ② 换月：用第 3 节的换月日历，换月时同方向移仓
- v3 = v2 + ③ 窗口：{v3_txt}。规则在跑第二轮之前写定
- v4 = v3 + ④ 方向性平仓：持空 z < 0.5、持多 z > −0.5 即平（原为 |z| < 0.5）

{mtable(fx_tab, ['version', 'fill'], '版本 / 成交')}

保守口径逐步变化：""" + '；'.join(steps) + f"""。{v34_txt}

![各版本净值](fig/equity_versions.png)

## 7 稳健性（只用第二轮样本内）

**参数网格**：在 v4 的规则上（冷静期、换月、方向性平仓），窗口 {{{', '.join(map(str, WINDOWS))}}} × 开仓 {{{', '.join(f'{e:g}' for e in ENTRIES)}}} × 平仓 {{{', '.join(f'{x:g}' for x in EXITS)}}}，止损 = 1.5 × 开仓阈值（沿用原脚本 `STD_THRESHOLD * 1.5` 的写法），1 跳成本。窗口 30 指过去 30 个交易日。

![夏普热力图](fig/heatmap_sharpe.png)

保守口径 36 格中夏普为正的有 {int((gi['sharpe'] > 0).sum())} 格，中位数 {num(gi['sharpe'].median())}，最高 {num(gi['sharpe'].max())}，最低 {num(gi['sharpe'].min())}；各格交易笔数 {int(gi['n'].min())}–{int(gi['n'].max())} 笔。{('选中格处在网格边上（' + '、'.join(edge) + '），邻域比内部格少，网格以外的参数没有测。') if edge else ''}

**选参规则**：每格与它在窗口、开仓、平仓三个方向上的相邻格（含对角，边界截断）取保守口径夏普的平均，选平均最高的格，不看单点最高。选出窗口 {P['window']}、开仓 {P['entry']:g}、平仓 {P['exit']:g}：本格夏普 {num(P['sharpe'])}，邻域平均 {num(P['neighborhood_sharpe'])}。单点最高的格是窗口 {P['best_single']['window']}、开仓 {P['best_single']['entry']:g}、平仓 {P['best_single']['exit']:g}，夏普 {num(P['best_single']['sharpe'])}，其邻域平均 {num(P['best_single']['neighborhood_sharpe'])}。参数写入 `report/round2/params.json` 并单独提交，之后才运行第二轮样本外。本次运行按同一规则重算的选择与冻结参数{'一致' if same else '不一致（样本外仍用冻结参数）'}。

**分年度表现**（冻结参数，保守口径，1 跳）：

{table(yr.rename(columns={'ret': '年度收益', 'sharpe': '夏普', 'mdd': '年内最大回撤', 'n': '开仓次数'}).rename_axis('年份'), {'年度收益': pct, '年内最大回撤': pct, '开仓次数': lambda v: f'{int(v)}'})}

样本内约 {R['is_years']:.1f} 年、{len(yr)} 个自然年度（首尾两年不完整），收益为正的有 {int((yr['ret'] > 0).sum())} 个；冻结参数样本内共 {int(pc.n)} 笔交易。

**成本敏感性**（冻结参数，样本内）：

{mtable(c_is, ['fill', 'ticks'], '成交 / 滑点跳数')}

**平稳性与半衰期**（样本内）：

| 序列 | ADF 统计量 | p 值 | 滞后阶数 | 样本数 | 半衰期（交易日） |
|---|---|---|---|---|---|
| 去换月跳空的连续价差 | {st['adj_stat']:.2f} | {st['adj_p']:.3f} | {st['adj_lags']} | {st['adj_n']} | {st['adj_hl']:.1f} |

每组合约在持期内单独检验（在持不少于 40 天的 {len(st['per'])} 组）：ADF p < 0.05 的有 {int((st['per']['p'] < 0.05).sum())} 组，p 值中位数 {st['per']['p'].median():.3f}；半衰期中位数 {st['per']['hl'].median():.1f} 天。

**价差的期限结构**：样本内 {ts['n']} 次换月中，新组合约的 S 低于旧组同日收盘的占 {pct(ts['gap_neg'])}，平均差 {ts['gap_mean']:,.0f} 元/单位。在持组逐日变化累加为 {st['adj'].iloc[-1]:+,.0f} 元，在持组原始 S 从 {ts['raw0']:,.0f} 元变为 {ts['raw1']:,.0f} 元。冻结参数样本内（保守口径）做空利润 {int(side.at[-1, 'size'])} 笔、净盈亏合计 {side.at[-1, 'sum']:,.0f} 元/单位；做多利润 {int(side.at[1, 'size'])} 笔、合计 {side.at[1, 'sum']:,.0f} 元/单位。

## 8 样本外（第二轮，{day(r.oos_start)} 至 {day(last)}，冻结参数；这段数据在第一轮已看过，只作参考）

参数：窗口 {P['window']}、开仓 {P['entry']:g}、平仓 {P['exit']:g}、止损 {P['entry'] * P['stop_mult']:g}，冷静期、换月（只在 01/05/09 里选）、方向性平仓均开启。样本外从空仓开始。

{mtable(c_oos, ['fill', 'ticks'], '成交 / 滑点跳数')}

保守口径、1 跳：第二轮样本外年化收益 {pct(po.ann_ret)}，夏普 {num(po.sharpe)}，最大回撤 {pct(po.mdd)}，{int(po.n)} 笔交易；同参数样本内夏普 {num(pc.sharpe)}；乐观口径样本外夏普 {num(pco.sharpe)}。{oos_cmp}

{gap_txt}

样本外交易（第二轮，保守口径、1 跳）：

{table(oos_tab, {'开仓z': num, '平仓z': num, '净盈亏(元/单位)': lambda v: f'{v:,.0f}'}) if len(oos_t) else '（无交易）'}

![样本外净值](fig/equity_oos.png)

## 9 实盘视角

**三腿同步成交**：{day(r.is_start)} 起三腿交易时段相同，但 MA 在郑商所，L、PP 在大商所，三腿无法用一张组合单成交，只能分腿下单，先成交的腿要承担其余腿的价格变动。日线数据量化不了盘中分腿的风险，只能给出两个参照：样本内在持组合约的价差从前一日收盘到次日开盘的跳动绝对值中位数 {lv['gap_med']:,.0f} 元/单位、90% 分位 {lv['gap_p90']:,.0f} 元，同期 {W0} 日滚动 σ 中位数 {lv['sd_med']:,.0f} 元；冻结参数下保守、乐观口径的夏普，样本内为 {num(pc.sharpe)} 和 {num(pco_is.sharpe)}，样本外为 {num(po.sharpe)} 和 {num(pco.sharpe)}。信号在收盘产生，实盘要按接近收盘的价格成交，只能在收盘前几分钟用盘中价格算信号并三腿同时下单，成交价介于两种口径之间，还要承担分腿风险。

**保证金**：按 {MARGIN:.0%} 计，1 单位保证金中位数约 {lv['margin_med']/1e4:.2f} 万元。冻结参数样本内（保守口径）单笔交易持仓期间最大浮亏占保证金的比例，中位数 {pct(-lv['mae_med'])}，最差一笔 {pct(-lv['mae_worst'])}。实盘还要留出交易所临近交割月上调保证金和期货公司加收的部分；换月规则让持仓不进入交割月前一个月，避开了临近交割的保证金上调。

**流动性**：样本内在持合约日成交量中位数 L {lv['vol_med']['L']:,.0f} 手、PP {lv['vol_med']['PP']:,.0f} 手、MA {lv['vol_med']['MA']:,.0f} 手；换月当天新组合约的成交量中位数 L {lv['vol_roll']['L']:,.0f}、PP {lv['vol_roll']['PP']:,.0f}、MA {lv['vol_roll']['MA']:,.0f} 手。原脚本的下单量（100 手 L、100 手 PP、300 手 MA）分别占上述日成交量中位数的 {pct(lv['demo_share']['L'], 2)}、{pct(lv['demo_share']['PP'], 2)}、{pct(lv['demo_share']['MA'], 2)}。成交量按 tushare 原始口径，交易所单边、双边统计口径未核实；日成交量只说明全天容量，开盘和夜盘开盘时点的盘口深度需要逐笔数据另行评估。

## 10 结论与局限

""", conclusions(R, R1, R3, v0c, fx, names, gi, P, pc, po, pco, q1, d_c, same_o, one_jump, edge), f"""
**局限**：
- 第二轮的三处修改在看过第一轮样本外之后做出，第二轮样本外不是独立检验；独立检验只有第一轮那一次。
- 第二轮样本内约 {R['is_years']:.1f} 年，冻结参数只有 {int(pc.n)} 笔交易；样本外约 {R['oos_years']:.1f} 年、{int(po.n)} 笔，统计意义有限。
- 只有日线，信号每天判断一次，盘中触发、分腿成交、冲击成本都没有模拟；滑点按固定跳数假设，没有盘口数据校准。
- 手续费按名义额万分之一估计，与各期货公司实际费率不同；成本敏感性表给出 0、1、2 跳的结果。
- 选参只看夏普一个指标，邻域取平均的规则本身也是一种选择；网格以外的参数没有测。
- 第三轮的滚动检验每个检验年末强平，跨年不留仓；金额止损只测了 {MONEY_STOP:.0%} 一个档位，手数上下限 [{SIZE_CAP[0]:g}, {SIZE_CAP[1]:g}] 也是事先定死的一组值。
- 手数按连续数值算，实盘 1 单位是 1 手 L + 1 手 PP + 3 手 MA，只能取整数倍，小资金按这个比例下单会有取整误差。
- 起点依据里，大商所两份通知的官网页面对程序抓取返回脚本页，内容据搜索引擎收录的官网摘要和期货公司全文转载确认；郑商所 2019-12 的通知只找到媒体转载。

{round3_section(R3)}

## 附：复现

```bash
pip install -r requirements.txt
python run_all.py        # 重算两轮全部表格、图和本报告，并核对第一轮结果与入库版本一致
pytest -q                # 数据层与回测引擎测试
```

文件：`src/data.py` 数据层与换月日历；`src/backtest.py` 回测引擎；`src/analysis.py` 两轮配置与各任务的计算；`src/report.py` 图与报告；`report/round1/`、`report/round2/` 分别存放两轮的冻结参数 `params.json`、原版交易 `trades_original.csv`、样本外交易 `trades_oos.csv`、换月记录 `roll_calendar.csv`、参数网格 `grid_is.csv` 和指标表。
"""]
    (REPORT / 'report.md').write_text(''.join(s))


def conclusions(R, R1, R3, v0c, fx, names, gi, P, pc, po, pco, q1, d_c, same, one_jump, edge):
    st, st1 = R['stat'], R1['stat']
    v0_1 = row(R1['orig'][0], scope='full_is', fill='open', ticks=1)
    pc1 = row(R1['cost_is'][0], fill='open', ticks=1)
    sh = [fx[n].sharpe for n in names]
    down = sh[-1] < sh[0]
    strong = st['adj_p'] < 0.05 <= st1['adj_p']
    weak = max(pc.sharpe, po.sharpe, pc1.sharpe, q1.sharpe) < 0.5
    out = [
        f"1. 原版逻辑在第二轮样本内（三腿交易时段一致之后）扣除 1 跳成本（保守口径）年化收益 {pct(v0c.ann_ret)}、夏普 {num(v0c.sharpe)}、"
        f"最大回撤 {pct(v0c.mdd)}；第一轮样本内（{day(R1['r'].is_start)} 起）同一口径夏普 {num(v0_1.sharpe)}。",
        f"2. 四处修正累加后（v4，保守口径）夏普由 {num(sh[0])} {'降' if down else '升'}至 {num(sh[-1])}，年化收益由 "
        f"{pct(fx[names[0]].ann_ret)} 变为 {pct(fx[names[-1]].ann_ret)}。"
        + (f"下降主要在①止损冷静期这一步（夏普 {num(sh[0])}→{num(sh[1])}，最大回撤 {pct(fx[names[0]].mdd)}→{pct(fx[names[1]].mdd)}）："
           f"原版止损后次日同向重开的 {d_c['n_reentry']} 笔交易合计净赚 {d_c['reentry_net']:,.0f} 元/单位，冷静期把它们去掉了。"
           if down and sh[1] < sh[0] and d_c['reentry_net'] > 0 else ''),
        f"3. 参数网格保守口径 36 格中 {int((gi['sharpe'] > 0).sum())} 格夏普为正、中位数 {num(gi['sharpe'].median())}；"
        f"按邻域平均选出的参数（窗口 {P['window']}、开仓 {P['entry']:g}、平仓 {P['exit']:g}）样本内夏普 {num(pc.sharpe)}，"
        f"只有 {int(pc.n)} 笔交易" + (f"，处在网格边上（{'、'.join(edge)}）" if edge else '') + "。",
        f"4. 样本外保守口径（1 跳）夏普 {num(po.sharpe)}、年化收益 {pct(po.ann_ret)}，乐观口径夏普 {num(pco.sharpe)}，共 {int(po.n)} 笔交易"
        + ("；两种口径的差距基本来自一次隔夜跳空（第 8 节）" if one_jump else '') + "。"
        + ("两轮样本外逐笔相同，这就是第一轮那次干净检验的结果。" if same
           else f"第一轮（唯一一次干净的检验）保守口径夏普 {num(q1.sharpe)}。"),
        f"5. 第二轮样本内去掉换月跳空的连续价差 ADF p 值 {st['adj_p']:.3f}、半衰期 {st['adj_hl']:.0f} 个交易日"
        f"（第一轮样本内为 {st1['adj_p']:.2f}、{st1['adj_hl']:.0f} 天）"
        + ("，三腿交易时段一致之后价差的均值回归证据明显更强；但 " if strong else "。")
        + ("z 分数规则两轮的样本内、样本外保守口径夏普都不到 0.5" + ("，样本外保守口径没有正收益" if po.ann_ret <= 0 else '')
           + "，现有证据不支持实盘。" if weak else "第二轮的样本外表现只能作参考，需要用之后的新数据检验。"),
    ]
    ag = R3['metrics']
    ag = ag[(ag.fill == 'open') & (ag.scope == 'all')].set_index('variant')
    best = ag['sharpe'].idxmax()
    out.append(
        f"6. 第三轮用滚动检验（{len(R3['folds']['year'].unique())} 个检验年，每年只用之前 {WF_TRAIN} 年选参）重做一遍："
        + '；'.join(f"{v} 夏普 {num(ag.loc[v, 'sharpe'])}±{num(ag.loc[v, 'se'])}、年化 {pct(ag.loc[v, 'ann_ret'])}、"
                    f"最大回撤 {pct(ag.loc[v, 'mdd'])}" for v in VARIANTS)
        + f"。最好的一档是 {best}，夏普仍在一个标准误之内包含 0。按波动定手数能同时改善夏普和回撤，"
          f"金额止损两头都变差。")
    return '\n'.join(out) + '\n'
