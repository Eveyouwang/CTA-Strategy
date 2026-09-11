"""任务 0 数据体检：python -m src.check_data 打印全部检查；run_all.py 复用 summary() 写进报告。"""
import importlib
import sys

import numpy as np
import pandas as pd

from .data import CONTRACTS, DAILY, L_TICK5_BEFORE, LEGS, load_raw, sha256


def summary():
    """三品种：合约数、日期范围、缺失值、每手吨数、最小变动价位、代码格式。"""
    daily, contracts = load_raw()
    d = daily.join(contracts.set_index('ts_code')[['fut_code']], on='ts_code')
    rows = []
    for leg in LEGS:
        c = contracts[contracts.fut_code == leg]
        x = d[d.fut_code == leg]
        rows.append({
            '品种': leg, '交易所': c['exchange'].iloc[0],
            '合约表合约数': len(c), '日线合约数': x['ts_code'].nunique(), '日线行数': len(x),
            '首日': x['trade_date'].min(), '末日': x['trade_date'].max(),
            '每手吨数(per_unit)': '/'.join(f'{v:g}' for v in sorted(c['per_unit'].unique())),
            'multiplier非空数': int(c['multiplier'].notna().sum()),
            '最小变动价位': '/'.join(sorted(c['quote_unit_desc'].unique())),
            '代码示例(ts_code/symbol)': f"{c['ts_code'].iloc[0]} / {c['symbol'].iloc[0]}",
            '收盘缺失': int(x['close'].isna().sum()), '开盘缺失': int(x['open'].isna().sum()),
            '持仓量缺失': int(x['oi'].isna().sum()), '成交量为0行数': int((x['vol'] == 0).sum()),
        })
    return pd.DataFrame(rows).set_index('品种')


def tick_evidence():
    """有成交的行里开高低收全是 5 的整数倍的比例，分 L_TICK5_BEFORE 前后；最小变动价位 1 元时约一成。"""
    daily, contracts = load_raw()
    d = daily.join(contracts.set_index('ts_code')[['fut_code']], on='ts_code')
    t = d[d['vol'] > 0]
    all5 = (t[['open', 'high', 'low', 'close']] % 5 == 0).all(axis=1)
    before = t['trade_date'] < L_TICK5_BEFORE
    return pd.DataFrame({'之前': [all5[(t.fut_code == l) & before].mean() for l in LEGS],
                         '之后': [all5[(t.fut_code == l) & ~before].mean() for l in LEGS]}, index=list(LEGS))


def fill_counts():
    """无成交日补价的行数：收盘价用结算价补、开盘价用补后的收盘价补。"""
    daily, _ = load_raw()
    return {'收盘用结算价补': int(daily['close'].isna().sum()), '开盘用收盘价补': int(daily['open'].isna().sum()),
            '总行数': len(daily)}


def main():
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('python', sys.version.split()[0])
    for m in ('pandas', 'numpy', 'matplotlib', 'statsmodels', 'pytest'):
        print(m, importlib.import_module(m).__version__)
    for p in (DAILY, CONTRACTS):
        print('sha256', p.name, sha256(p))
    daily, contracts = load_raw()
    print('\n== 三品种概况 ==')
    print(summary().T.to_string())
    print('\n== 日线各列缺失值 ==')
    print(daily.isna().sum().to_string())

    d = daily.join(contracts.set_index('ts_code')[['fut_code', 'd_month', 'list_date', 'delist_date']],
                   on='ts_code')
    print('\n== 两个交易所交易日是否一致（2014-06-17 起）==')
    ex = {e: set(d.loc[(d.fut_code.isin(f)) & (d.trade_date >= '20140617'), 'trade_date'])
          for e, f in (('CZCE', ['MA']), ('DCE', ['L', 'PP']))}
    print('只在郑商所:', sorted(ex['CZCE'] - ex['DCE'])[:10], '只在大商所:', sorted(ex['DCE'] - ex['CZCE'])[:10])

    print('\n== 合约存续期内缺行（按全样本交易日历）==')
    cal = np.sort(d['trade_date'].unique())
    g = d.groupby('ts_code')['trade_date'].agg(['min', 'max', 'count'])
    g['expected'] = np.searchsorted(cal, g['max'], side='right') - np.searchsorted(cal, g['min'])
    g['gap'] = g['expected'] - g['count']
    print('有缺行的合约数:', int((g['gap'] > 0).sum()), '缺行合计:', int(g['gap'].sum()))
    print(g[g['gap'] > 0].sort_values('gap', ascending=False).head(10).to_string())
    print('首行日期晚于上市日的合约数:',
          int((g['min'] > contracts.set_index('ts_code').loc[g.index, 'list_date']).sum()))

    print('\n== 成交量为 0 的行：价格字段长什么样 ==')
    z = d[d['vol'] == 0]
    print(z.groupby('fut_code').size().to_string())
    print(z[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'settle', 'pre_close', 'vol', 'oi']]
          .head(8).to_string())
    print('\n== 收盘价 <= 0 的行数 ==', int((d['close'] <= 0).sum()))
    print('\n== 收盘价能被 5 整除的比例（核对最小变动价位是否为 1）==')
    print(d.assign(m5=d['close'] % 5 == 0).groupby('fut_code')['m5'].mean().round(3).to_string())
    print('\n== 同一合约同一天重复行 ==', int(d.duplicated(['ts_code', 'trade_date']).sum()))
    print(f'\n== 有成交行开高低收全是 5 的整数倍的比例（{L_TICK5_BEFORE} 前后）==')
    print(tick_evidence().round(3).to_string())
    print('\n== 无成交日补价 ==', fill_counts())


if __name__ == '__main__':
    main()
