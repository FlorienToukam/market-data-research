"""Independent checksum, SQL, arithmetic and model-result audits."""
from pathlib import Path
import hashlib
import importlib.metadata as im
import json
import py_compile
import sqlite3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results'
OUT.mkdir(parents=True, exist_ok=True)
(ROOT/'07_Environment').mkdir(parents=True, exist_ok=True)

def run():
    checks=[]
    manifest=json.loads((ROOT/'01_Data/source_manifest.json').read_text())
    plan_hash=hashlib.sha256((ROOT/'03_Code/research_plan.json').read_bytes()).hexdigest()
    assert plan_hash==manifest['plan_sha256']
    checks.append({'check':'Research plan unchanged since data download','result':'PASS'})
    for entry in manifest['files']:
        p=ROOT/entry['file'];data=p.read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['sha256'] and len(data)==entry['bytes']
        checks.append({'check':f'Source checksum: {p.name}','result':'PASS'})
    for file in (ROOT/'03_Code').glob('*.py'):py_compile.compile(str(file),doraise=True)
    checks.append({'check':'All research scripts compile','result':'PASS'})
    feature=pd.read_csv(ROOT/'01_Data/processed/optiver_five_minute_features.csv')
    prediction=pd.read_csv(OUT/'optiver_reserved_test_predictions.csv')
    summary=json.loads((OUT/'optiver_summary.json').read_text())
    assert len(feature)==summary['eligible_stock_buckets'] and len(prediction)==summary['test_selected']['observations']
    assert set(prediction.time_id).isdisjoint(set(feature.loc[feature.split!='test','time_id']))
    scores=pd.read_csv(OUT/'optiver_model_scores.csv')
    selected=summary['selected_model']
    validation=scores[(scores.split=='validation')&scores.model.isin(['Ridge','Gradient boosting'])]
    assert selected==validation.sort_values('rmspe').iloc[0].model
    for row in scores[scores.split=='test'].to_dict('records'):
        col='prediction_'+row['model'].lower().replace(' ','_')
        pred=prediction[col].to_numpy();observed=prediction.rv_target.to_numpy()
        # Explicit scalar summation is independent of sklearn's scorer.
        reference=(sum(((float(a)-float(y))/float(y))**2 for a,y in zip(pred,observed))/len(observed))**.5
        assert abs(reference-row['rmspe'])<1e-10
    checks.append({'check':'Reserved model scores independently recalculated and validation-only choice confirmed','result':'PASS'})
    assert summary['total_book_rows']==sum(pq.ParquetFile(p).metadata.num_rows for p in (ROOT/'01_Data/raw/optiver').glob('*_book.parquet'))
    assert summary['total_trade_rows']==sum(pq.ParquetFile(p).metadata.num_rows for p in (ROOT/'01_Data/raw/optiver').glob('*_trade.parquet'))
    checks.append({'check':'Raw record counts agree with parquet metadata','result':'PASS'})
    order_checks=[]
    for p in (ROOT/'01_Data/raw/optiver').glob('*_book.parquet'):
        b=pd.read_parquet(p)
        assert (b[['bid_size1','bid_size2','ask_size1','ask_size2']]>=0).all().all()
        assert (b.bid_price2<=b.bid_price1).all() and (b.ask_price2>=b.ask_price1).all()
        order_checks.append(p.name)
    checks.append({'check':'Order-book price levels and nonnegative sizes','result':'PASS','files':order_checks})
    db=ROOT/'01_Data/processed/market_research.sqlite'
    with sqlite3.connect(db) as con:
        feature.to_sql('optiver_features',con,if_exists='replace',index=False)
        frames=[]
        for p in (ROOT/'01_Data/raw/etf').glob('*_daily_prices.csv'):
            f=pd.read_csv(p);f['ticker']=p.name.split('_')[0];frames.append(f)
        pd.concat(frames,ignore_index=True).to_sql('etf_prices',con,if_exists='replace',index=False)
        pd.read_csv(OUT/'SPY_simulated_trade_log.csv').to_sql('spy_trade_log',con,if_exists='replace',index=False)
        queries={
            'duplicate_feature_keys':'SELECT COUNT(*) FROM (SELECT stock_id,time_id FROM optiver_features GROUP BY stock_id,time_id HAVING COUNT(*)>1)',
            'split_leakage_groups':'SELECT COUNT(*) FROM (SELECT time_id FROM optiver_features GROUP BY time_id HAVING COUNT(DISTINCT split)>1)',
            'invalid_volatility_rows':'SELECT COUNT(*) FROM optiver_features WHERE rv_first IS NULL OR rv_target IS NULL OR rv_first<=0 OR rv_target<=0',
            'duplicate_etf_dates':'SELECT COUNT(*) FROM (SELECT ticker,date FROM etf_prices GROUP BY ticker,date HAVING COUNT(*)>1)'}
        query_results={name:int(con.execute(sql).fetchone()[0]) for name,sql in queries.items()}
        assert all(v==0 for v in query_results.values())
        pd.read_sql_query('SELECT stock_id,split,COUNT(*) AS stock_buckets,AVG(spread_bps) AS mean_quote_spread_bps FROM optiver_features GROUP BY stock_id,split ORDER BY stock_id,split',con).to_csv(OUT/'sql_sample_summary.csv',index=False)
    (OUT/'sql_validation.json').write_text(json.dumps(query_results,indent=2))
    checks.append({'check':'SQLite independent duplicate, split and missing-data checks','result':'PASS','results':query_results})
    perf=pd.read_csv(OUT/'strategy_performance.csv')
    strategy_summary=json.loads((OUT/'strategy_summary.json').read_text())
    for ticker in ['SPY','QQQ','IWM']:
        for name in ['Trend filter','Buy and hold','50% ETF / 50% cash']:
            filename=f'{ticker}_{name.lower().replace(" ","_").replace("/","_")}_daily_ledger.csv'
            ledger=pd.read_csv(OUT/filename,parse_dates=['date']).set_index('date')
            previous=np.r_[100000,ledger.wealth.to_numpy()[:-1]]
            np.testing.assert_allclose(ledger.wealth.to_numpy(),previous*(1+ledger.net_return),rtol=1e-12)
            held=ledger.loc['2020-01-01':'2026-09-30'];r=held.net_return.to_numpy()
            wealth=1.;high=1.;drawdown=0.
            for daily in r:
                wealth*=1+float(daily);high=max(high,wealth);drawdown=min(drawdown,wealth/high-1)
            ref=perf[(perf.ticker==ticker)&(perf.strategy==name)&(perf.period=='Held-out')].iloc[0]
            assert abs(wealth-1-ref.total_return)<1e-10 and abs(drawdown-ref.max_drawdown)<1e-10
            assert abs((wealth**(252/len(r))-1)-ref.cagr)<1e-10
    checks.append({'check':'All nine strategy/benchmark ledgers and holdout CAGR/drawdown independently reconciled','result':'PASS'})
    results={'status':'PASS','checks':checks,'completed_checks':len(checks),
             'note':'Calculation checks are separate from the methodological limitations documented in the report.'}
    (OUT/'verification_report.json').write_text(json.dumps(results,indent=2))
    packages=['numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow','reportlab','pypdf']
    versions={p:im.version(p) for p in packages}
    (ROOT/'07_Environment/package_versions.json').write_text(json.dumps(versions,indent=2))
    (ROOT/'03_Code/requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'sql':query_results,'versions':versions},indent=2),flush=True)
    return results

if __name__=='__main__':run()
