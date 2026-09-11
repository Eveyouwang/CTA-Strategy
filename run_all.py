"""一键重算全部结果：数据校验 → 原版 → 逐项修正 → 稳健性 → 样本外（冻结参数）→ 报告。

python run_all.py
样本外只读取 report/params.json 的冻结参数，不在这里重新选参。
"""
import json
import time

import pandas as pd

from src import analysis as A
from src.check_data import fill_counts, summary, tick_evidence
from src.data import CONTRACTS, DAILY, sha256
from src.report import write

SHA256 = {  # 任务 0 记录的 data/ 文件指纹
    'daily_MA_L_PP.csv': '5ed7d15fc3d3562c1eb798efeac4621b90838935bd12f2155347c6647eaf8341',
    'contracts_MA_L_PP.csv': 'ac2452795c1953f682f8103963a99c3a15d9d296ddcb6290b1f31a4c80b0b8df',
}


def main():
    t0 = time.time()
    sha = {p.name: sha256(p) for p in (DAILY, CONTRACTS)}
    assert sha == SHA256, f'data/ 与任务 0 记录的 sha256 不一致：{sha}'
    lab = A.Lab()
    R = dict(lab=lab, sha=sha, summary=summary(), ticks=tick_evidence(), fills=fill_counts())
    R['stat'] = A.stationarity(lab)
    R['w3'] = A.window_from_half_life(R['stat']['adj_hl'])
    R['w_hl'] = int(round(R['stat']['adj_hl']))
    R['cov_hl'] = A.z_coverage(lab, R['w_hl'])
    R['orig'] = A.original(lab)
    R['defects'] = A.defects(lab, R['orig'][2], R['orig'][1])
    R['fix'] = A.fixes(lab, R['w3'])
    R['grid'] = A.grid(lab)
    R['sel'] = A.select(R['grid'])
    R['params'] = json.loads(A.PARAMS.read_text())
    R['cost_is'] = A.costs(lab, R['params'], A.IS_START, A.IS_END)
    R['yearly'] = A.yearly(*R['cost_is'][1]['open', 1])
    R['oos'] = A.costs(lab, R['params'], A.OOS_START, lab.mkt.dates[-1])
    R['oos_years'] = len(R['oos'][1]['open', 1][0]) / 252
    R['live'] = A.live(lab, R['cost_is'][1])
    R['rolls'] = A.roll_stats(lab)

    out = A.REPORT
    R['orig'][1].to_csv(out / 'trades_original.csv', index=False)
    pd.concat([t.assign(fill=f, ticks=k) for (f, k), (_, t) in R['oos'][1].items()]).to_csv(
        out / 'trades_oos.csv', index=False)
    R['rolls']['table'].to_csv(out / 'roll_calendar.csv', index=False)
    R['grid'].to_csv(out / 'grid_is.csv', index=False)
    R['orig'][0].to_csv(out / 'metrics_original.csv', index=False)
    R['fix'][0].to_csv(out / 'metrics_fixes.csv', index=False)
    R['oos'][0].to_csv(out / 'metrics_oos.csv', index=False)
    write(R)

    cols = ['ann_ret', 'mdd', 'sharpe', 'win', 'n', 'hold']
    pd.set_option('display.width', 200)
    print('== 原版 ==\n', R['orig'][0][['scope', 'fill', 'ticks'] + cols].round(3).to_string(index=False))
    print('== 修正版 ==\n', R['fix'][0][['version', 'fill'] + cols].round(3).to_string(index=False))
    print('== 样本外 ==\n', R['oos'][0][['fill', 'ticks'] + cols].round(3).to_string(index=False))
    print(f'report/report.md 已生成，耗时 {time.time() - t0:.1f} 秒')


if __name__ == '__main__':
    main()
