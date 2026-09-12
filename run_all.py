"""一键重算全部结果：数据校验 → 第一轮（原设计，重算并核对入库结果）→ 第二轮（原版 → 逐项修正 → 稳健性 → 样本外）→ 报告。

python run_all.py
两轮的样本外都只读取各自 report/roundN/params.json 的冻结参数，不在这里重新选参。
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src import analysis as A
from src.check_data import fill_counts, summary, tick_evidence
from src.data import CONTRACTS, DAILY, load_market, sha256
from src.report import write

SHA256 = {  # 任务 0 记录的 data/ 文件指纹
    'daily_MA_L_PP.csv': '5ed7d15fc3d3562c1eb798efeac4621b90838935bd12f2155347c6647eaf8341',
    'contracts_MA_L_PP.csv': 'ac2452795c1953f682f8103963a99c3a15d9d296ddcb6290b1f31a4c80b0b8df',
}
ROUND1_SHA256 = {  # 第一轮首次运行时入库的冻结参数与样本外结果，重算必须逐字节一致
    'params.json': 'f84c3a26c4acddb329ed99b9dcbe1e61ef31a269d8fec58a2d6fb8886d051a73',
    'metrics_oos.csv': 'dc29649a5ef6e2cfbcf0ced029209699cf35931f9d9694eb9998beec67314fb6',
}
# 交易明细在第三轮多了一列手数（前两轮恒为 1）；去掉该列后必须与首次运行入库的文件一致
ROUND1_TRADES_SHA256 = '21ad6d3f000b2c9582563af5ffba8333d2f80fb1ad466984510e47c0e409d7bb'


def run_round(r, mkt):
    lab = A.Lab(r, mkt)
    R = dict(lab=lab, r=r)
    R['stat'] = A.stationarity(lab)
    hl = R['stat']['adj_hl']
    R['w_hl'] = int(round(hl)) if np.isfinite(hl) else None
    R['w_hist'] = A.history_limit(lab)
    R['w3'] = A.window3(lab, hl)
    R['cov_hl'] = A.z_coverage(lab, R['w_hl']) if R['w_hl'] else None
    R['orig'] = A.original(lab)
    R['defects'] = A.defects(lab, R['orig'][2], R['orig'][1])
    R['fix'] = A.fixes(lab, R['w3'])
    R['grid'] = A.grid(lab)
    R['sel'] = A.select(R['grid'], r)
    R['params'] = json.loads(r.params.read_text())
    R['cost_is'] = A.costs(lab, R['params'], r.is_start, r.is_end)
    R['yearly'] = A.yearly(*R['cost_is'][1]['open', 1])
    R['oos'] = A.costs(lab, R['params'], r.oos_start, mkt.dates[-1])
    R['is_years'] = len(R['cost_is'][1]['open', 1][0]) / 252
    R['oos_years'] = len(R['oos'][1]['open', 1][0]) / 252
    R['live'] = A.live(lab, R['cost_is'][1])
    R['rolls'] = A.roll_stats(lab)
    R['ts'] = A.term_structure(lab)

    out = r.out
    out.mkdir(parents=True, exist_ok=True)
    R['orig'][1].to_csv(out / 'trades_original.csv', index=False)
    R['trades_oos'] = pd.concat([t.assign(fill=f, ticks=k) for (f, k), (_, t) in R['oos'][1].items()])
    R['trades_oos'].to_csv(out / 'trades_oos.csv', index=False)
    R['rolls']['table'].to_csv(out / 'roll_calendar.csv', index=False)
    R['grid'].to_csv(out / 'grid_is.csv', index=False)
    R['orig'][0].to_csv(out / 'metrics_original.csv', index=False)
    R['fix'][0].to_csv(out / 'metrics_fixes.csv', index=False)
    R['oos'][0].to_csv(out / 'metrics_oos.csv', index=False)
    return R


def run_wf(mkt):
    """第三轮：滚动检验（每个检验年只用它之前 3 年选参）× 三档风控。"""
    lab = A.Lab(A.ROUND2, mkt)
    vol = A.spread_vol(lab)
    folds, runs, rows = [], {}, []
    for v in A.VARIANTS:
        f, r = A.walk_forward(lab, v, vol)
        folds.append(f)
        runs[v] = r
        for scope, since in (('all', None), ('since2020', '2020')):
            for fill, m in A.wf_summary(f, r, since=since).items():
                rows.append(dict(variant=v, scope=scope, fill=fill, **m))
    R = dict(lab=lab, folds=pd.concat(folds, ignore_index=True), runs=runs, metrics=pd.DataFrame(rows))
    out = A.REPORT / 'round3'
    out.mkdir(parents=True, exist_ok=True)
    R['folds'].to_csv(out / 'folds.csv', index=False)
    R['metrics'].to_csv(out / 'metrics_wf.csv', index=False)
    pd.concat([runs[v]['open'][1] for v in A.VARIANTS]).to_csv(out / 'trades_wf.csv', index=False)
    pd.DataFrame({v: 1 + runs[v]['open'][0]['ret'].cumsum() for v in A.VARIANTS}).to_csv(out / 'nav_wf.csv')
    return R


def main():
    t0 = time.time()
    sha = {p.name: sha256(p) for p in (DAILY, CONTRACTS)}
    assert sha == SHA256, f'data/ 与任务 0 记录的 sha256 不一致：{sha}'
    mkt = load_market()
    R1 = run_round(A.ROUND1, mkt)
    got = {n: sha256(A.ROUND1.out / n) for n in ROUND1_SHA256}
    assert got == ROUND1_SHA256, f'第一轮重算结果与首次运行入库的不一致：{got}'
    old = hashlib.sha256(R1['trades_oos'].drop(columns=['qty']).to_csv(index=False).encode()).hexdigest()
    assert old == ROUND1_TRADES_SHA256, f'第一轮样本外交易明细（去掉手数列）与入库的不一致：{old}'
    R2 = run_round(A.ROUND2, mkt)
    R3 = run_wf(mkt)
    write(R2, R1, R3, dict(sha=sha, summary=summary(), ticks=tick_evidence(), fills=fill_counts()))

    cols = ['ann_ret', 'mdd', 'sharpe', 'win', 'n', 'hold']
    pd.set_option('display.width', 200)
    print('第一轮重算与入库结果逐字节一致：', ', '.join(ROUND1_SHA256), '、trades_oos.csv（去掉手数列）')
    for R in (R1, R2):
        r = R['r']
        print(f'===== {r.name}：样本内 {r.is_start}–{r.is_end}，窗口③ {R["w3"]} 天，冻结参数 '
              f'{R["params"]["window"]}/{R["params"]["entry"]}/{R["params"]["exit"]} =====')
        print('== 原版 ==\n', R['orig'][0][['scope', 'fill', 'ticks'] + cols].round(3).to_string(index=False))
        print('== 修正版 ==\n', R['fix'][0][['version', 'fill'] + cols].round(3).to_string(index=False))
        print('== 样本外 ==\n', R['oos'][0][['fill', 'ticks'] + cols].round(3).to_string(index=False))
    print('===== round3：滚动检验（保守口径、1 跳、本金 3 倍保证金）=====')
    m = R3['metrics']
    print(m[m.fill == 'open'][['variant', 'scope', 'ann_ret', 'mdd', 'sharpe', 'se', 't', 'n', 'win']].round(3).to_string(index=False))
    print(f'report/report.md 已生成，耗时 {time.time() - t0:.1f} 秒')


if __name__ == '__main__':
    main()
