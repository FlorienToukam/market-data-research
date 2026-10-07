"""Download immutable public-source snapshots; never execute remote source code."""
from pathlib import Path
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import time
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / '03_Code/research_plan.json').read_text())
RAW = ROOT / '01_Data/raw'
RAW.mkdir(parents=True, exist_ok=True)

def get(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 FlorienAcademicMarketResearch'})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

def record(path, url, **extra):
    data = path.read_bytes()
    return {'file': str(path.relative_to(ROOT)), 'url': url,
            'retrieved_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), **extra}

def download_optiver():
    cfg = PLAN['optiver']
    url = f"https://api.github.com/repos/{cfg['mirror_repo']}/git/trees/{cfg['mirror_commit']}?recursive=1"
    tree = json.loads(get(url))
    (RAW / 'optiver').mkdir(exist_ok=True)
    (RAW / 'optiver/mirror_tree.json').write_text(json.dumps(tree, indent=2))
    wanted = []
    for entry in tree['tree']:
        path = entry['path']
        for stock in cfg['stock_ids']:
            for kind in ['book', 'trade']:
                if path.startswith(f'{kind}_train.parquet/stock_id={stock}/') and path.endswith('.parquet'):
                    wanted.append((entry, stock, kind))
    assert len(wanted) == 2 * len(cfg['stock_ids']), 'Incomplete book/trade source pairs'
    def fetch(item):
        entry, stock, kind = item
        url = f"https://raw.githubusercontent.com/{cfg['mirror_repo']}/{cfg['mirror_commit']}/{entry['path']}"
        path = RAW / f'optiver/stock_{stock}_{kind}.parquet'
        data = get(url)
        blob_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert blob_sha == entry['sha'], 'Git source integrity mismatch'
        assert len(data) == entry['size'], 'Truncated download'
        assert data[:4] == b'PAR1' and data[-4:] == b'PAR1', 'Not a parquet file'
        path.write_bytes(data)
        print(f'Downloaded Optiver stock {stock} {kind}: {len(data):,} bytes', flush=True)
        return record(path, url, source='Optiver Kaggle public data via third-party GitHub mirror',
                      stock_id=stock, kind=kind, git_blob_sha1=blob_sha,
                      original_source='https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data')
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(fetch, wanted))

def download_etf(ticker):
    cfg = PLAN['strategy']
    epoch = lambda s: int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp())
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={epoch(cfg["download_start"])}'
           f'&period2={epoch(cfg["download_end_exclusive"])}&interval=1d&events=div%2Csplits')
    payload = get(url)
    obj = json.loads(payload)
    assert obj['chart']['error'] is None
    result = obj['chart']['result'][0]
    assert result['meta']['symbol'] == ticker
    folder = RAW / 'etf'
    folder.mkdir(exist_ok=True)
    raw = folder / f'{ticker}_yahoo_snapshot.json'
    raw.write_bytes(payload)
    quote = result['indicators']['quote'][0]
    frame = pd.DataFrame(quote)
    frame['date'] = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert('America/New_York').date
    frame['adjusted_close'] = result['indicators']['adjclose'][0]['adjclose']
    frame = frame[['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
    frame.to_csv(folder / f'{ticker}_daily_prices.csv', index=False)
    events = []
    for kind, event_list in result.get('events', {}).items():
        for event in event_list.values():
            events.append({'kind': kind, **event})
    (folder / f'{ticker}_corporate_actions.json').write_text(json.dumps(events, indent=2))
    print(f'Downloaded {ticker}: {len(frame):,} daily observations', flush=True)
    return record(raw, url, source='Yahoo Finance chart API snapshot', ticker=ticker,
                  first_date=str(frame.date.min()), last_date=str(frame.date.max()), rows=len(frame),
                  caveat='Vendor-adjusted prices may be revised; cached raw snapshot is the reproducibility reference')

def main():
    records = download_optiver()
    cfg = PLAN['strategy']
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        records += list(pool.map(download_etf, [cfg['primary_ticker'], *cfg['transfer_checks']]))
    (ROOT / '01_Data/source_manifest.json').write_text(json.dumps({'plan_sha256': hashlib.sha256((ROOT / '03_Code/research_plan.json').read_bytes()).hexdigest(), 'files': records}, indent=2))
    print('Data snapshot and checksum manifest saved.', flush=True)

if __name__ == '__main__':
    main()
