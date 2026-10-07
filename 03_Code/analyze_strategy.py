"""Delayed-close, self-financing long/cash ETF research. No parameter optimization."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'03_Code/research_plan.json').read_text())
CFG=PLAN['strategy'];OUT=ROOT/'04_Results';CHARTS=ROOT/'05_Charts'
OUT.mkdir(exist_ok=True);CHARTS.mkdir(exist_ok=True)

def read_prices(ticker):
    p=pd.read_csv(ROOT/f'01_Data/raw/etf/{ticker}_daily_prices.csv',parse_dates=['date']).set_index('date').sort_index()
    assert p.index.is_unique and p.index.is_monotonic_increasing
    assert p.notna().all().all() and (p[['open','high','low','close','adjusted_close']]>0).all().all()
    assert (p.volume>0).all() and (p.high>=p.low).all()
    assert (p.high+1e-5>=p[['open','close']].max(axis=1)).all()
    assert (p.low-1e-5<=p[['open','close']].min(axis=1)).all()
    assert (p.index.dayofweek<5).all() and str(p.index[-1].date())==CFG['evaluation_end']
    p['asset_return']=p.adjusted_close.pct_change()
    events=json.loads((ROOT/f'01_Data/raw/etf/{ticker}_corporate_actions.json').read_text())
    div={}
    for event in events:
        if event['kind']=='dividends':
            date=pd.to_datetime(event['date'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
            div[date]=float(event['amount'])
    dividends=pd.Series(div,dtype=float).reindex(p.index,fill_value=0)
    simple_total=(p.close+dividends)/p.close.shift()-1
    difference=(p.asset_return-simple_total).abs()
    non_div=difference[dividends==0].dropna()
    assert non_div.max()<1e-4, 'Unexpected adjustment on a non-dividend date'
    qc={'ticker':ticker,'rows':len(p),'start':str(p.index[0].date()),'end':str(p.index[-1].date()),
        'null_cells':int(p.drop(columns='asset_return').isna().sum().sum()),'duplicate_dates':0,
        'dividend_events_in_range':int((dividends>0).sum()),
        'max_non_dividend_adjusted_return_difference_bps':float(non_div.max()*10000),
        'max_dividend_adjusted_vs_simple_total_return_difference_bps':float(difference[dividends>0].max()*10000)}
    return p,qc

def targets(prices,lookback=200,delay=1,fixed_weight=None):
    if fixed_weight is not None:
        s=pd.Series(float(fixed_weight),index=prices.index)
    else:
        ma=prices.close.rolling(lookback,min_periods=lookback).mean()
        signal=(prices.close>ma).astype(float).where(ma.notna(),0.0)
        s=signal.shift(delay,fill_value=0.0)
    return s.loc[CFG['evaluation_start']:CFG['evaluation_end']]

def simulate(prices,lookback=200,delay=1,cost_bps=5,cash_annual=0.0,fixed_weight=None,liquidate=True):
    target=targets(prices,lookback,delay,fixed_weight)
    dates=target.index
    r=prices.loc[dates,'asset_return'].to_numpy(float)
    c=(1+cash_annual)**(1/252)-1
    a=target.to_numpy(float)
    held=np.r_[0.0,a[:-1]]
    k=cost_bps/10000
    factor=1+held*r+(1-held)*c
    drift=held*(1+r)/factor
    # Exact one-way cost paid on traded notional, with final holdings at the target weight.
    fraction=np.where(a>=drift,k*(a-drift)/(1+k*a),k*(drift-a)/(1-k*a))
    close_target=a.copy()
    if liquidate:
        fraction[-1]=k*drift[-1]
        close_target[-1]=0.0
    gross_trade=np.abs(close_target*(1-fraction)-drift)
    net=factor*(1-fraction)-1
    wealth=CFG['initial_capital']*np.cumprod(1+net)
    before=np.r_[CFG['initial_capital'],wealth[:-1]]*factor
    frame=pd.DataFrame({'asset_return':r,'weight_held_during_session':held,
                        'weight_after_close':close_target,'pre_trade_weight':drift,
                        'turnover_fraction':gross_trade,'fee_fraction':fraction,
                        'fee_dollars':before*fraction,'gross_return':factor-1,
                        'net_return':net,'wealth':wealth},index=dates)
    frame['close_signal_price']=prices.loc[dates,'close']
    frame['adjusted_close_return_proxy']=prices.loc[dates,'adjusted_close']
    frame['trailing_sma']=prices.close.rolling(lookback,min_periods=lookback).mean().reindex(dates)
    frame['signal_at_today_close']=(frame.close_signal_price>frame.trailing_sma).astype(int)
    frame.index.name='date'
    assert np.isfinite(frame.drop(columns='trailing_sma').to_numpy()).all() and (frame.fee_dollars>=-1e-10).all()
    assert ((frame.weight_held_during_session>=0)&(frame.weight_held_during_session<=1)).all()
    return frame

def independent_ledger(prices,target,cost_bps=5,cash_annual=0.0):
    # Independent dollar-account calculation with fractional units of the adjusted-price return proxy.
    # Uses cash and units, not the vectorized return/weight formula above.
    cash=float(CFG['initial_capital']); units=0.0;k=cost_bps/10000
    c=(1+cash_annual)**(1/252)-1; records=[]
    for j,(date,w) in enumerate(target.items()):
        price=float(prices.loc[date,'adjusted_close']);cash*=1+c
        asset=units*price;before=cash+asset
        if j==len(target)-1:w=0.0
        intended=w*before
        if intended>=asset: trade=(intended-asset)/(1+k*w)
        else: trade=(intended-asset)/(1-k*w)
        fee=abs(trade)*k
        units+=trade/price;cash-=trade+fee
        after=cash+units*price
        records.append((date,after,fee))
    return pd.DataFrame(records,columns=['date','wealth','fee_dollars']).set_index('date')

def metrics(frame):
    r=frame.net_return;wealth=(1+r).cumprod()
    high=np.maximum.accumulate(np.r_[1.0,wealth.to_numpy()])[1:]
    dd=wealth.to_numpy()/high-1
    std=float(r.std(ddof=1));n=len(r)
    return {'observations':n,'cagr':float(wealth.iloc[-1]**(252/n)-1),
            'total_return':float(wealth.iloc[-1]-1),'annualized_volatility':std*np.sqrt(252),
            'sharpe_zero_hurdle':float(r.mean()/std*np.sqrt(252)) if std else None,
            'max_drawdown':float(dd.min()),'average_exposure':float(frame.weight_held_during_session.mean()),
            'one_way_turnover':float(frame.turnover_fraction.sum()),
            'fees_dollars':float(frame.fee_dollars.sum()),
            'entry_count':int(((frame.weight_after_close>0)&(frame.weight_held_during_session==0)).sum()),
            'exit_count':int(((frame.weight_after_close==0)&(frame.weight_held_during_session>0)).sum())}

def verify(prices,frame,ticker):
    original=targets(prices)
    date=original.index[len(original)//2]
    altered=prices.copy();altered.loc[altered.index>date,'close']*=1.4
    changed=targets(altered)
    pd.testing.assert_series_equal(original.loc[:date],changed.loc[:date],check_exact=True)
    ledger=independent_ledger(prices,original)
    np.testing.assert_allclose(frame.wealth,ledger.wealth,rtol=1e-11,atol=1e-5)
    np.testing.assert_allclose(frame.fee_dollars,ledger.fee_dollars,rtol=1e-9,atol=1e-6)
    for w in [.5,1.0]:
        bench=simulate(prices,fixed_weight=w)
        check=independent_ledger(prices,targets(prices,fixed_weight=w))
        np.testing.assert_allclose(bench.wealth,check.wealth,rtol=1e-11,atol=1e-5)
        np.testing.assert_allclose(bench.fee_dollars,check.fee_dollars,rtol=1e-9,atol=1e-6)
    lagged=(prices.close>prices.close.rolling(200).mean()).shift(2,fill_value=False).astype(float).reindex(frame.index)
    np.testing.assert_array_equal(frame.weight_held_during_session.iloc[1:],lagged.iloc[1:])
    # Scalar toy example verifies signal at session 1 enters at session 2 close, earns from session 3.
    toy=pd.DataFrame({'close':[100.,101.,103.,110.,108.],'adjusted_close':[100.,101.,103.,110.,108.]},
                     index=pd.date_range('2002-01-02',periods=5))
    toy['asset_return']=toy.adjusted_close.pct_change().fillna(0)
    tf=simulate(toy,lookback=2,delay=1,cost_bps=0,liquidate=False)
    assert list(tf.weight_held_during_session)==[0.,0.,0.,1.,1.]
    assert abs(tf.net_return.iloc[2])<1e-12 and abs(tf.net_return.iloc[3]-(110/103-1))<1e-12
    return {'ticker':ticker,'future_price_perturbation':'PASS','independent_cash_units_ledger':'PASS',
            'independent_benchmark_ledgers':'PASS',
            'signal_execution_return_lag':'PASS','toy_execution_fixture':'PASS',
            'max_ledger_wealth_difference_dollars':float((frame.wealth-ledger.wealth).abs().max())}

def run():
    primary={};rows=[];quality=[];checks=[];yearly=[]
    for ticker in [CFG['primary_ticker'],*CFG['transfer_checks']]:
        prices,qc=read_prices(ticker);quality.append(qc)
        choices={'Trend filter':simulate(prices),
                 'Buy and hold':simulate(prices,fixed_weight=1),
                 '50% ETF / 50% cash':simulate(prices,fixed_weight=.5)}
        checks.append(verify(prices,choices['Trend filter'],ticker))
        for name,frame in choices.items():
            frame.to_csv(OUT/f'{ticker}_{name.lower().replace(" ","_").replace("/","_")}_daily_ledger.csv')
            for period,dates in [('Full history',[CFG['evaluation_start'],CFG['evaluation_end']]),
                                 ('Development',CFG['development']),('Validation',CFG['validation']),('Held-out',CFG['held_out'])]:
                rows.append({'ticker':ticker,'strategy':name,'period':period,
                             'start':str(frame.loc[dates[0]:dates[1]].index[0].date()),
                             'end':str(frame.loc[dates[0]:dates[1]].index[-1].date()),**metrics(frame.loc[dates[0]:dates[1]])})
            if ticker==CFG['primary_ticker']:
                for year,g in frame.groupby(frame.index.year):
                    yearly.append({'year':year,'strategy':name,'return':float((1+g.net_return).prod()-1)})
        if ticker==CFG['primary_ticker']:primary=choices
        print(f'{ticker}: delayed strategy, both benchmarks, independent ledger and causality checks complete.',flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'strategy_performance.csv',index=False)
    events=primary['Trend filter'][primary['Trend filter'].turnover_fraction>1e-8].copy()
    events['action']=np.where(events.weight_after_close>events.weight_held_during_session,'BUY','SELL')
    events['execution_scope']='Modeled next-session-close transaction, not an actual execution'
    events.to_csv(OUT/'SPY_simulated_trade_log.csv')
    pd.DataFrame(quality).to_csv(OUT/'etf_data_quality.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'strategy_verification.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'SPY_calendar_returns.csv',index=False)
    prices,_=read_prices(CFG['primary_ticker']);sens=[]
    for lookback in CFG['sensitivity_lookbacks']:
        for cost in CFG['sensitivity_one_way_cost_bps']:
            f=simulate(prices,lookback=lookback,cost_bps=cost)
            sens.append({'lookback':lookback,'cost_bps':cost,'delay':1,'cash_annual':0.,'kind':'Lookback / cost',**metrics(f.loc[CFG['held_out'][0]:])})
    for delay in [2,3]:
        f=simulate(prices,delay=delay)
        sens.append({'lookback':200,'cost_bps':5,'delay':delay,'cash_annual':0.,'kind':'Execution delay',**metrics(f.loc[CFG['held_out'][0]:])})
    f=simulate(prices,cash_annual=.03)
    sens.append({'lookback':200,'cost_bps':5,'delay':1,'cash_annual':.03,'kind':'Illustrative cash yield',**metrics(f.loc[CFG['held_out'][0]:])})
    pd.DataFrame(sens).to_csv(OUT/'strategy_sensitivity.csv',index=False)
    # Moving-block bootstrap of mean excess daily returns, preserving short-range serial dependence.
    excess=(primary['Trend filter'].net_return-primary['Buy and hold'].net_return).loc[CFG['held_out'][0]:].to_numpy()
    rng=np.random.default_rng(2026);block=20;n=len(excess);boot=[]
    for _ in range(1000):
        starts=rng.integers(0,n-block+1,int(np.ceil(n/block)))
        draw=np.concatenate([excess[i:i+block] for i in starts])[:n]
        boot.append(float(draw.mean()*252))
    main=result[(result.ticker==CFG['primary_ticker'])&(result.period=='Held-out')]
    summary={'primary_ticker':CFG['primary_ticker'],'held_out_start':main.iloc[0].start,'held_out_end':main.iloc[0].end,
             'held_out_metrics':{r['strategy']:{k:v for k,v in r.items() if k not in ['ticker','strategy','period','start','end']} for r in main.to_dict('records')},
             'annualized_mean_excess_return_95_block_bootstrap_ci':np.quantile(boot,[.025,.975]).tolist(),
             'block_bootstrap_block_sessions':block,'annualization':252,
             'rule':'200-session price SMA, delayed one session, 5 bps one-way cost, zero cash interest',
             'evaluation_type':'Historical chronological research holdout; not prospective live testing',
             'parameter_selection':'No parameter selected using held-out results'}
    (OUT/'strategy_summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'Trend filter':'#2563eb','Buy and hold':'#475569','50% ETF / 50% cash':'#0f766e'}
    fig,axes=plt.subplots(2,1,figsize=(9,6.2),sharex=True)
    for name,frame in primary.items():
        held=frame.loc[CFG['held_out'][0]:];nav=(1+held.net_return).cumprod()
        draw=nav/np.maximum.accumulate(np.r_[1,nav.to_numpy()])[1:]-1
        axes[0].plot(nav.index,nav*100,label=name,color=colors[name],lw=1.5)
        axes[1].plot(nav.index,draw*100,color=colors[name],lw=1)
    axes[0].set(title='SPY: held-out historical performance, net of modeled trading costs',ylabel='Growth of $100')
    axes[0].legend(fontsize=9);axes[1].set(ylabel='Drawdown (%)');fig.tight_layout()
    fig.savefig(CHARTS/'SPY_held_out_performance.png',dpi=180);plt.close(fig)
    s=pd.DataFrame(sens);chart=s[(s.kind=='Lookback / cost')&(s.cost_bps==5)]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(chart.lookback.astype(str),chart.cagr*100,color=['#94a3b8','#94a3b8','#2563eb','#94a3b8'])
    ax.axhline(main[main.strategy=='Buy and hold'].iloc[0].cagr*100,color='#475569',ls='--',label='Buy-and-hold CAGR')
    ax.set(xlabel='SMA lookback (sessions)',ylabel='Held-out CAGR (%)',title='Robustness check: primary rule stays at 200 sessions')
    ax.legend();fig.tight_layout();fig.savefig(CHARTS/'SPY_parameter_sensitivity.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':run()
