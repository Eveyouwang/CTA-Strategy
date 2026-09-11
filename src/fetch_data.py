"""一次性下载 data/ 下的两个原始文件（run_all.py 不调用本脚本）。

用法：TUSHARE_TOKEN=<token> TUSHARE_URL=<接口地址> python -m src.fetch_data
- contracts_MA_L_PP.csv：fut_basic(fut_type=1) 返回的 MA(郑商所)、L、PP(大商所) 全部普通合约，原样保存
- daily_MA_L_PP.csv：上述合约中最后交易日 >= 2014-01-01 的逐合约 fut_daily 日线，原样保存
  （PP 2014-02 上市、10 吨/手的 MA 2014-06 上市，更早的 L 合约用不上）
下载完成后 data/ 只读，两个文件的 sha256 记在 PROGRESS.md。token 不入库。
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
END = '20260911'
PRODUCTS = [('CZCE', 'MA'), ('DCE', 'L'), ('DCE', 'PP')]


def query(api, **params):
    body = json.dumps({'api_name': api, 'token': os.environ['TUSHARE_TOKEN'],
                       'params': params, 'fields': ''}).encode()
    # 代理拒绝 urllib 默认 UA（实测 403），沿用 tushare SDK 所用 requests 的 UA
    req = urllib.request.Request(f"{os.environ['TUSHARE_URL']}/{api}", data=body,
                                 headers={'Content-Type': 'application/json',
                                          'User-Agent': 'python-requests/2.32.3'})
    for k in range(6):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.URLError as e:  # 代理偶发 SSL 断连（实测第 41 个合约时出现），等几秒重试
            print('retry', api, params.get('ts_code'), e, flush=True)
            time.sleep(5 * (k + 1))
            continue
        if r['code'] == 0:
            return pd.DataFrame(r['data']['items'], columns=r['data']['fields'])
        if '每分钟' not in r['msg']:  # 只有限频才等待重试，其余错误直接抛出
            raise RuntimeError(f"{api} {params}: {r['msg']}")
        time.sleep(61)
    raise RuntimeError(f'{api} {params}: 重试 6 次仍失败')


def main():
    (ROOT / 'data').mkdir(exist_ok=True)
    basic = pd.concat([query('fut_basic', exchange=ex, fut_type='1', fut_code=fc)
                       for ex, fc in PRODUCTS], ignore_index=True)
    basic.to_csv(ROOT / 'data/contracts_MA_L_PP.csv', index=False)
    todo = basic[basic['delist_date'] >= '20140101'].sort_values('ts_code')
    frames = []
    for i, c in enumerate(todo.itertuples(), 1):
        d = query('fut_daily', ts_code=c.ts_code, start_date=c.list_date, end_date=END)
        assert len(d) < 2000, c.ts_code  # fut_daily 单次上限 2000 行，逐合约远低于此
        frames.append(d)
        print(i, len(todo), c.ts_code, len(d), flush=True)
        time.sleep(0.3)
    daily = pd.concat(frames, ignore_index=True).sort_values(['ts_code', 'trade_date'])
    daily.to_csv(ROOT / 'data/daily_MA_L_PP.csv', index=False)
    print('done', basic.shape, daily.shape)


if __name__ == '__main__':
    main()
