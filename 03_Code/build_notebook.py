"""Create a readable notebook, without inventing past notebook execution history."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def markdown_table(df):
    headers=list(df.columns)
    rows=['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']
    rows += ['| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def main():
    out=ROOT/'02_Notebooks';out.mkdir(exist_ok=True)
    scores=pd.read_csv(ROOT/'04_Results/optiver_model_scores.csv')
    scores=scores[scores.split=='test'][['model','rmspe','rmse','mae']].copy()
    scores['rmspe']=scores.rmspe.map(lambda x:f'{x:.2%}')
    scores['rmse']=scores.rmse.map(lambda x:f'{x:.6f}')
    scores['mae']=scores.mae.map(lambda x:f'{x:.6f}')
    perf=pd.read_csv(ROOT/'04_Results/strategy_performance.csv')
    perf=perf[(perf.period=='Held-out')&(perf.ticker=='SPY')][['strategy','cagr','annualized_volatility','max_drawdown']].copy()
    for col in perf.columns[1:]:perf[col]=perf[col].map(lambda x:f'{x:.2%}')
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'id':f'cell-{len(cells)+1:03d}','metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','# Market Data Research and Trading Strategy Evaluation\n\nFlorien Siakoua Toukam | Reconstructed October 7, 2026\n\nThis read-and-rerun notebook includes verified saved-result tables below. Code cells have not been assigned fictional prior execution counts. Run them with the project dependencies to reproduce the cached-data analysis. The two experiments are separate: the volatility model does not drive the ETF trades.')
    add('code',"from pathlib import Path\nimport sys, json\nimport pandas as pd\nROOT = Path.cwd()\nif not (ROOT / '03_Code').exists():\n    ROOT = ROOT.parent\nassert (ROOT / '03_Code/research_plan.json').exists()\nsys.path.insert(0, str(ROOT / '03_Code'))\nplan = json.loads((ROOT / '03_Code/research_plan.json').read_text())\nplan")
    add('markdown','## 1. Source snapshots and integrity\n\nSix Optiver parquet files cover anonymized stocks 0, 1 and 3. Yahoo snapshots retain SPY, QQQ and IWM prices and corporate actions. The original request URLs, raw byte counts and SHA-256 hashes are recorded in the manifest.')
    add('code',"manifest = json.loads((ROOT / '01_Data/source_manifest.json').read_text())\npd.DataFrame(manifest['files'])[['file','bytes','source','sha256']]")
    add('markdown','## 2. Five-minute features and prediction target\n\nFirst-half seconds 0-299 provide all predictors. Second-half seconds 300-599 provide realized volatility to predict. This is a reconstructed five-minute task, not the original Kaggle target. Complete shuffled time buckets stay in one split, but the test is not chronological.')
    add('code',"import analyze_optiver\noptiver_summary = analyze_optiver.run()\nfeatures = pd.read_csv(ROOT / '01_Data/processed/optiver_five_minute_features.csv')\nfeatures.groupby(['stock_id','split']).size().unstack()")
    add('markdown','### Saved reserved-test results\n\n'+markdown_table(scores)+'\n\nThe full Ridge gain is 16.2% versus the naive baseline, 3.4% versus calibration, and 4.3% versus RV-only Ridge. The confidence interval versus calibration includes zero. Forecasting variability is not forecasting direction or trading profit.')
    add('code',"pd.read_csv(ROOT / '04_Results/optiver_model_scores.csv')")
    add('markdown','![Five-minute volatility model comparison](../05_Charts/optiver_model_comparison.png)\n\nCompare the validation and test results, then inspect the model coefficients. The selected full model is determined by validation RMSPE, not by picking the best test result.')
    add('code',"pd.read_csv(ROOT / '04_Results/optiver_ridge_coefficients.csv')")
    add('markdown','## 3. Delayed trading-strategy evaluation\n\nThe fixed primary rule goes long when SPY closes above its 200-session price average and otherwise holds cash. A signal executes at the next session close and earns returns only thereafter. Costs are 5 bps on one-way traded notional. Main cash interest is zero. Historical later-period evaluation is 2020-September 2026.\n\n### Saved SPY later-period results\n\n'+markdown_table(perf))
    add('code',"import analyze_strategy\nstrategy_summary = analyze_strategy.run()\npd.read_csv(ROOT / '04_Results/strategy_performance.csv').query(\"period == 'Held-out'\")")
    add('markdown','![SPY later-period performance](../05_Charts/SPY_held_out_performance.png)\n\nThe rule reduces drawdown and volatility but gives up return. The half-stock/half-cash benchmark also shows how lower exposure can reduce risk. The model is not selected by searching the held-out lookbacks.')
    add('code',"pd.read_csv(ROOT / '04_Results/strategy_sensitivity.csv')")
    add('markdown','## 4. Follow an execution\n\nThe saved simulated trade log can show a BUY on a day whose newly observed close signal is CASH. That is consistent with following the previous session decision. Inspect March 2020 whipsaws rather than assuming a delayed trade uses today\'s information.')
    add('code',"trades = pd.read_csv(ROOT / '04_Results/SPY_simulated_trade_log.csv')\ntrades.query(\"date >= '2020-02-01' and date <= '2020-06-30'\")")
    add('markdown','## 5. Independent verification and SQL\n\nSource checksums, raw record counts, non-overlapping splits, future-information perturbation tests, independent portfolio ledgers, and scalar metric recalculations are checked. SQLite provides a separate processed-data audit. See the report for the methodological limitations that remain even when calculation checks pass.')
    add('code',"import verify_project\nverification = verify_project.run()\nverification['status'], verification['completed_checks']")
    add('code',"import sqlite3\nwith sqlite3.connect(ROOT / '01_Data/processed/market_research.sqlite') as con:\n    summary = pd.read_sql_query('SELECT stock_id, split, COUNT(*) AS n FROM optiver_features GROUP BY stock_id, split', con)\nsummary")
    add('markdown','## Conclusion\n\nThe project supports discussion of market-data preparation, short-horizon risk forecasting, validation, execution timing, trading costs and investment judgment. It does not establish a live profitable trading system. Read the study guide before adapting the case into resume language.')
    obj={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    path=out/'Market_Research_Analysis.ipynb';path.write_text(json.dumps(obj,indent=2))
    assert len(json.loads(path.read_text())['cells'])==len(cells)
    print('Notebook saved:',len(cells),'cells',flush=True)

if __name__=='__main__':main()
