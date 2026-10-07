"""Five-minute feature/target research with a time-bucket-grouped holdout."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / '03_Code/research_plan.json').read_text())
OUT = ROOT / '04_Results'
CHARTS = ROOT / '05_Charts'
PROCESSED = ROOT / '01_Data/processed'
for p in [OUT, CHARTS, PROCESSED]: p.mkdir(parents=True, exist_ok=True)

def realized_volatility(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(np.diff(np.log(values)) ** 2))) if len(values) > 1 else 0.0

def extract(book, trade, stock):
    # No quote or trade at second 300 or later enters a predictor.
    book = book.sort_values(['time_id', 'seconds_in_bucket']).copy()
    for col in ['bid_price1','ask_price1','bid_price2','ask_price2']:
        book[col] = book[col].astype('float64')
    den = book.bid_size1 + book.ask_size1
    book['wap'] = (book.bid_price1 * book.ask_size1 + book.ask_price1 * book.bid_size1) / den
    book['spread_bps'] = (book.ask_price1 - book.bid_price1) / ((book.ask_price1 + book.bid_price1) / 2) * 10000
    book['depth'] = den
    book['imbalance'] = (book.bid_size1 - book.ask_size1) / den
    book['abs_imbalance'] = book.imbalance.abs()
    first = book[book.seconds_in_bucket < 300]
    second = book[book.seconds_in_bucket >= 300]
    f = first.groupby('time_id').agg(
        rv_first=('wap', realized_volatility), spread_bps=('spread_bps', 'mean'),
        depth=('depth', 'mean'), imbalance=('imbalance', 'mean'),
        abs_imbalance=('abs_imbalance', 'mean'), quote_observations=('wap', 'size'),
        first_min_second=('seconds_in_bucket', 'min'), first_max_second=('seconds_in_bucket', 'max'))
    target = second.groupby('time_id').agg(
        rv_target=('wap', realized_volatility), target_observations=('wap', 'size'),
        target_min_second=('seconds_in_bucket', 'min'), target_max_second=('seconds_in_bucket', 'max'))
    t = trade[trade.seconds_in_bucket < 300].groupby('time_id').agg(
        trade_volume=('size', 'sum'), trade_records=('size', 'size'), execution_count=('order_count', 'sum'))
    f = f.join(t).fillna({'trade_volume': 0, 'trade_records': 0, 'execution_count': 0})
    f = f.join(target)
    f['stock_id'] = stock
    return f.reset_index()

NUMERIC = ['log_rv_first', 'log_spread_bps', 'log_depth', 'imbalance', 'abs_imbalance',
           'log_quote_observations', 'log_trade_volume', 'log_trade_records', 'log_execution_count']

def add_model_features(frame):
    frame = frame.copy()
    frame['log_rv_first'] = np.log(frame.rv_first.clip(lower=1e-8))
    for x in ['spread_bps', 'depth', 'quote_observations', 'trade_volume', 'trade_records', 'execution_count']:
        frame['log_' + x] = np.log1p(frame[x])
    return frame

def score(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert (y > 0).all() and np.isfinite(prediction).all() and (prediction > 0).all()
    return {'rmspe': float(np.sqrt(np.mean(((prediction-y)/y)**2))),
            'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'mae': float(mean_absolute_error(y, prediction)),
            'spearman': float(spearmanr(y, prediction).statistic), 'observations': len(y)}

def run():
    frames, quality = [], []
    for stock in PLAN['optiver']['stock_ids']:
        folder = ROOT / '01_Data/raw/optiver'
        book = pd.read_parquet(folder / f'stock_{stock}_book.parquet')
        trade = pd.read_parquet(folder / f'stock_{stock}_trade.parquet')
        qc = {'stock_id': stock, 'book_rows': len(book), 'trade_rows': len(trade),
              'book_duplicates': int(book.duplicated(['time_id', 'seconds_in_bucket']).sum()),
              'trade_duplicates': int(trade.duplicated(['time_id', 'seconds_in_bucket']).sum()),
              'book_null_cells': int(book.isna().sum().sum()), 'trade_null_cells': int(trade.isna().sum().sum()),
              'crossed_quotes': int((book.bid_price1 > book.ask_price1).sum()),
              'nonpositive_prices': int((book[['bid_price1','ask_price1','bid_price2','ask_price2']] <= 0).any(axis=1).sum()),
              'invalid_depth': int(((book.bid_size1+book.ask_size1) <= 0).sum()),
              'invalid_trade_size': int((trade['size'] <= 0).sum()),
              'invalid_seconds': int((~book.seconds_in_bucket.between(0,599)).sum() + (~trade.seconds_in_bucket.between(0,599)).sum())}
        assert all(qc[k] == 0 for k in ['book_duplicates','trade_duplicates','book_null_cells','trade_null_cells','crossed_quotes','nonpositive_prices','invalid_depth','invalid_trade_size','invalid_seconds']), qc
        f = extract(book, trade, stock)
        qc['available_buckets'] = len(f)
        eligible = ((f.quote_observations >= 10) & (f.target_observations >= 10) &
                    (f.first_min_second <= 60) & (f.first_max_second >= 240) &
                    (f.target_min_second <= 360) & (f.target_max_second >= 540) &
                    (f.rv_first > 0) & (f.rv_target > 0))
        qc['eligible_buckets'] = int(eligible.sum())
        qc['excluded_buckets'] = int((~eligible).sum())
        if (~eligible).any():
            f.loc[~eligible].to_csv(OUT / f'optiver_excluded_buckets_stock_{stock}.csv', index=False)
        # Future perturbation test: changing second-half quotes must not alter first-half predictors.
        changed = book.copy()
        future = changed.seconds_in_bucket >= 300
        perturbation = 1 + (changed.loc[future, 'seconds_in_bucket'] - 300) * .0003
        for col in ['bid_price1','ask_price1','bid_price2','ask_price2']:
            changed[col] = changed[col].astype('float64')
            changed.loc[future, col] *= perturbation
        changed.loc[future, ['bid_size1','ask_size1']] *= 3
        changed_trade = trade.copy()
        changed_trade.loc[changed_trade.seconds_in_bucket >= 300, 'size'] *= 3
        f_changed = extract(changed, changed_trade, stock)
        predictor_cols = ['time_id', 'rv_first', 'spread_bps', 'depth', 'imbalance', 'abs_imbalance',
                          'quote_observations', 'trade_volume', 'trade_records', 'execution_count']
        pd.testing.assert_frame_equal(f[predictor_cols], f_changed[predictor_cols], check_exact=True)
        assert not np.allclose(f.rv_target.fillna(0), f_changed.rv_target.fillna(0))
        qc['future_quote_perturbation_test'] = 'PASS'
        frames.append(f[eligible].copy()); quality.append(qc)
        print(f'Optiver stock {stock}: {len(book):,} quotes, {len(trade):,} trades; {eligible.sum():,} eligible buckets', flush=True)

    data = add_model_features(pd.concat(frames, ignore_index=True))
    groups = np.array(sorted(data.time_id.unique()))
    rng = np.random.default_rng(PLAN['optiver']['seed']); rng.shuffle(groups)
    n_train, n_val = int(len(groups)*.70), int(len(groups)*.15)
    sets = {'train': set(groups[:n_train]), 'validation': set(groups[n_train:n_train+n_val]), 'test': set(groups[n_train+n_val:])}
    assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
    data['split'] = data.time_id.map({g: split for split, values in sets.items() for g in values})
    data.to_csv(PROCESSED / 'optiver_five_minute_features.csv', index=False)
    # Every stock sharing a bucket must stay on the same side of the split.
    assert data.groupby('time_id').split.nunique().max() == 1
    train = data[data.split == 'train']; validation = data[data.split == 'validation']; test = data[data.split == 'test']
    prep_ridge = ColumnTransformer([('numeric', StandardScaler(), NUMERIC),
                                    ('stock', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['stock_id'])])
    prep_tree = ColumnTransformer([('numeric', 'passthrough', NUMERIC),
                                   ('stock', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['stock_id'])])
    models = {
        'Ridge': make_pipeline(prep_ridge, Ridge(alpha=PLAN['optiver']['ridge_alpha'])),
        'Gradient boosting': make_pipeline(prep_tree, HistGradientBoostingRegressor(random_state=42, **PLAN['optiver']['hist_gradient_boosting']))}
    rows = []
    ratio = train.rv_first.to_numpy() / train.rv_target.to_numpy()
    calibration = float(ratio.sum() / np.square(ratio).sum())
    reduced_prep = ColumnTransformer([('rv', StandardScaler(), ['log_rv_first']),
                                      ('stock', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['stock_id'])])
    reduced_model = make_pipeline(reduced_prep, Ridge(alpha=1.0))
    reduced_model.fit(train[['log_rv_first','stock_id']], np.log(train.rv_target))
    predictions = {}
    for split, subset in [('validation',validation),('test',test)]:
        predictions[split] = {'Persistence': subset.rv_first.to_numpy(),
                              'Calibrated persistence': subset.rv_first.to_numpy()*calibration,
                              'RV-only Ridge': np.exp(reduced_model.predict(subset[['log_rv_first','stock_id']]))}
    for name, model in models.items():
        model.fit(train[NUMERIC+['stock_id']], np.log(train.rv_target))
        for split, subset in [('validation', validation), ('test', test)]:
            predictions[split][name] = np.exp(model.predict(subset[NUMERIC+['stock_id']]))
    for split, subset in [('validation', validation), ('test', test)]:
        for name, pred in predictions[split].items(): rows.append({'split': split, 'model': name, **score(subset.rv_target, pred)})
    scores = pd.DataFrame(rows)
    selected = scores[(scores.split == 'validation') & scores.model.isin(models)].sort_values('rmspe').iloc[0].model
    # Models remain fitted on training only. Validation chooses a model, not new parameters.
    test = test.copy()
    for name, pred in predictions['test'].items(): test['prediction_' + name.lower().replace(' ', '_')] = pred
    test.to_csv(OUT / 'optiver_reserved_test_predictions.csv', index=False)
    scores.to_csv(OUT / 'optiver_model_scores.csv', index=False)
    pd.DataFrame(quality).to_csv(OUT / 'optiver_data_quality.csv', index=False)
    per_stock = []
    for stock, subset in test.groupby('stock_id'):
        for name in ['Persistence', selected]:
            per_stock.append({'stock_id': stock, 'model': name,
                              **score(subset.rv_target, subset['prediction_' + name.lower().replace(' ', '_')])})
    pd.DataFrame(per_stock).to_csv(OUT / 'optiver_test_scores_by_stock.csv', index=False)
    # Group bootstrap, not row bootstrap, keeps simultaneous stock observations together.
    model_col = 'prediction_' + selected.lower().replace(' ', '_')
    test['baseline_squared_pct_error'] = ((test.prediction_persistence-test.rv_target)/test.rv_target)**2
    test['model_squared_pct_error'] = ((test[model_col]-test.rv_target)/test.rv_target)**2
    test['calibrated_squared_pct_error'] = ((test.prediction_calibrated_persistence-test.rv_target)/test.rv_target)**2
    test['reduced_squared_pct_error'] = ((test['prediction_rv-only_ridge']-test.rv_target)/test.rv_target)**2
    grouped = test.groupby('time_id')[['baseline_squared_pct_error','model_squared_pct_error','calibrated_squared_pct_error','reduced_squared_pct_error']].agg(['sum','count'])
    bsum = grouped[('baseline_squared_pct_error','sum')].to_numpy()
    msum = grouped[('model_squared_pct_error','sum')].to_numpy()
    counts = grouped[('model_squared_pct_error','count')].to_numpy()
    csum = grouped[('calibrated_squared_pct_error','sum')].to_numpy()
    rsum = grouped[('reduced_squared_pct_error','sum')].to_numpy()
    rng = np.random.default_rng(2026); improvements=[]; calibrated_improvements=[]; reduced_improvements=[]
    for _ in range(1000):
        ix=rng.integers(0,len(grouped),len(grouped)); n=counts[ix].sum()
        b=np.sqrt(bsum[ix].sum()/n); m=np.sqrt(msum[ix].sum()/n)
        improvements.append((b-m)/b)
        c=np.sqrt(csum[ix].sum()/n); r=np.sqrt(rsum[ix].sum()/n)
        calibrated_improvements.append((c-m)/c); reduced_improvements.append((r-m)/r)
    baseline = score(test.rv_target, test.prediction_persistence)
    chosen = score(test.rv_target, test[model_col])
    calibrated = score(test.rv_target, test.prediction_calibrated_persistence)
    reduced = score(test.rv_target, test['prediction_rv-only_ridge'])
    ridge_features = models['Ridge'].steps[0][1].get_feature_names_out()
    pd.DataFrame({'feature':ridge_features,'standardized_log_target_coefficient':models['Ridge'].steps[-1][1].coef_}).to_csv(OUT/'optiver_ridge_coefficients.csv',index=False)
    summary = {'task': PLAN['optiver']['task'], 'stock_ids': PLAN['optiver']['stock_ids'],
               'total_book_rows': sum(q['book_rows'] for q in quality),
               'total_trade_rows': sum(q['trade_rows'] for q in quality),
               'eligible_stock_buckets': len(data), 'unique_time_buckets': len(groups),
               'excluded_stock_buckets': sum(q['excluded_buckets'] for q in quality),
               'split_rows': data.split.value_counts().to_dict(),
               'split_groups': {k:len(v) for k,v in sets.items()}, 'selected_model': selected,
               'test_baseline': baseline, 'test_selected': chosen,
               'test_calibrated_persistence': calibrated, 'test_rv_only_ridge': reduced,
               'persistence_calibration_train_only': calibration,
               'relative_improvement_vs_calibrated_persistence': (calibrated['rmspe']-chosen['rmspe'])/calibrated['rmspe'],
               'relative_improvement_vs_rv_only_ridge': (reduced['rmspe']-chosen['rmspe'])/reduced['rmspe'],
               'relative_rmspe_improvement': (baseline['rmspe']-chosen['rmspe'])/baseline['rmspe'],
               'group_bootstrap_improvement_95_ci': np.quantile(improvements,[.025,.975]).tolist(),
               'group_bootstrap_calibrated_improvement_95_ci': np.quantile(calibrated_improvements,[.025,.975]).tolist(),
               'group_bootstrap_rv_only_improvement_95_ci': np.quantile(reduced_improvements,[.025,.975]).tolist(),
               'holdout_type': 'Random unseen time-bucket groups; calendar ordering unavailable',
               'causality': 'Predictive association; no causal liquidity claim and no trading profit claim'}
    (OUT / 'optiver_summary.json').write_text(json.dumps(summary, indent=2))
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax=plt.subplots(figsize=(9,4.2))
    t=scores[scores.split=='test']; ax.bar(t.model.str.replace(' ', '\n'),t.rmspe*100,color=['#94a3b8','#64748b','#94a3b8','#2563eb','#0f766e'])
    ax.set_ylabel('RMSPE (%) - lower is better'); ax.set_title('Reserved bucket-group test: five-minute volatility')
    for i,v in enumerate(t.rmspe*100): ax.text(i,v+1,f'{v:.1f}%',ha='center')
    fig.tight_layout(); fig.savefig(CHARTS/'optiver_model_comparison.png',dpi=180);plt.close(fig)
    fig, ax=plt.subplots(figsize=(7,4.2))
    ax.scatter(test.rv_target*100,test[model_col]*100,s=8,alpha=.3,color='#2563eb')
    limit=float(max(test.rv_target.max(),test[model_col].max())*100)
    ax.plot([0,limit],[0,limit],color='#475569',lw=1)
    ax.set(xlabel='Observed second-five-minute volatility (%)',ylabel='Predicted volatility (%)',
           title=f'{selected}: prediction versus observed risk')
    fig.tight_layout();fig.savefig(CHARTS/'optiver_prediction_scatter.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__': run()
