"""Build the research memorandum and study guide from verified result files."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'06_Reports';REPORTS.mkdir(exist_ok=True)
O=json.loads((ROOT/'04_Results/optiver_summary.json').read_text())
S=json.loads((ROOT/'04_Results/strategy_summary.json').read_text())
V=json.loads((ROOT/'04_Results/verification_report.json').read_text())
PERF=pd.read_csv(ROOT/'04_Results/strategy_performance.csv')
SCORES=pd.read_csv(ROOT/'04_Results/optiver_model_scores.csv')
SENS=pd.read_csv(ROOT/'04_Results/strategy_sensitivity.csv')
M=S['held_out_metrics'];T=M['Trend filter'];B=M['Buy and hold'];F=M['50% ETF / 50% cash']
NAVY=colors.HexColor('#14243a');BLUE=colors.HexColor('#2563eb');GREY=colors.HexColor('#526071')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCase',fontName='Helvetica-Bold',fontSize=24,leading=28,textColor=NAVY,spaceAfter=12))
styles.add(ParagraphStyle(name='SubtitleCase',fontName='Helvetica',fontSize=11,leading=15,textColor=GREY,spaceAfter=15))
styles.add(ParagraphStyle(name='SectionCase',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=NAVY,spaceBefore=10,spaceAfter=8))
styles.add(ParagraphStyle(name='BodyCase',fontName='Helvetica',fontSize=10.2,leading=14.2,textColor=NAVY,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallCase',fontName='Helvetica',fontSize=8.2,leading=11,textColor=GREY,spaceAfter=6))
styles.add(ParagraphStyle(name='CellCase',fontName='Helvetica',fontSize=8.8,leading=11.5,textColor=NAVY))
styles.add(ParagraphStyle(name='HeadCellCase',fontName='Helvetica-Bold',fontSize=8.7,leading=11,textColor=colors.white))

def p(text,style='BodyCase'):return Paragraph(text,styles[style])
def heading(text):return p(text,'SectionCase')
def percent(v,d=2):return f'{100*v:.{d}f}%'
def table(rows,widths):
    cells=[[p(escape(str(c)),'HeadCellCase' if i==0 else 'CellCase') for c in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                          ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),
                          ('LINEBELOW',(0,0),(-1,0),.5,NAVY)]))
    return [t,Spacer(1,10)]
def chart(name,width=490,height=None):
    from PIL import Image as PILImage
    path=ROOT/'05_Charts'/name
    with PILImage.open(path) as im:w,h=im.size
    return Image(str(path),width=width,height=height or width*h/w,hAlign='LEFT')
def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
    canvas.line(48,44,564,44);canvas.setFont('Helvetica',8);canvas.setFillColor(GREY)
    canvas.drawString(48,31,'Florien Siakoua Toukam | Market Research Case Study | October 7, 2026')
    canvas.drawRightString(564,31,str(doc.page));canvas.restoreState()
def build(path,story,title):
    doc=SimpleDocTemplate(str(path),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=59,
                          title=title,author='Florien Siakoua Toukam',subject='Public-data research case, reconstructed October 2026')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def memorandum():
    a=[p('Market Data Research and<br/>Trading Strategy Evaluation','TitleCase'),
       p('Optiver order-book research and an ETF trend-filter backtest<br/>Research memorandum | Florien Siakoua Toukam | October 7, 2026','SubtitleCase'),
       heading('Investment research question'),
       p('Can order-book and trade information improve a short-horizon volatility forecast, and can a simple trading rule reduce market risk without giving up too much return? I examined these questions separately because the available datasets support different tests.'),
       heading('What the analysis shows')]
    a+=table([['Research test','Result','Implication'],
              ['Five-minute volatility',f'Ridge test RMSPE {percent(O["test_selected"]["rmspe"])} versus {percent(O["test_baseline"]["rmspe"])} for persistence.','The model improved the basic forecast. Stronger baselines substantially narrow the gain.'],
              ['SPY trend filter, 2020-2026',f'Net CAGR {percent(T["cagr"])} versus {percent(B["cagr"])} for buy-and-hold.','The rule reduced risk but surrendered return in the later historical period.'],
              ['SPY downside',f'Maximum drawdown {percent(T["max_drawdown"])} versus {percent(B["max_drawdown"])}.','Risk reduction is meaningful, although a 50% stock/cash benchmark had a smaller drawdown.']], [122,169,225])
    a += [heading('Conclusion'),
          p('I would treat the volatility model as a risk-estimation input for further research. Its advantage over a calibrated simple forecast is too uncertain to describe as a durable edge. I would treat the trend filter as an exposure-management rule whose return cost must be justified by the investor\'s risk objective.'),
          p('The strongest finding is the distinction between reducing risk and improving investment returns. Neither a better volatility forecast nor a lower drawdown, by itself, establishes a profitable trading strategy.'),
          p('Scope: a newly reconstructed academic research case using public historical data. The simulations are not professional trading, actual executions, or work performed for Optiver or SG. ETF test period ends September 30, 2026; 2026 is partial.','SmallCase'),PageBreak(),
          heading('1. Order-book and trade research'),
          p(f'I processed {O["total_book_rows"]:,} quote records and {O["total_trade_rows"]:,} trade records for anonymized stock IDs 0, 1 and 3. The six parquet files are saved with source URLs, a fixed mirror commit, and checksums. Stock identities and calendar dates are not supplied.'),
          p('For each observed ten-minute bucket, I used seconds 0-299 to build predictors and seconds 300-599 to measure the target. The target is second-half realized volatility based on top-of-book weighted-average-price log changes. This is a custom five-minute task, not the competition\'s original subsequent-ten-minute target.'),
          p('Predictors cover prior realized volatility, relative bid-ask spread, displayed depth, order-book imbalance, quote-observation count, trade volume, trade-record count and execution count. Numeric inputs are transformed where appropriate; Ridge scaling is fitted on training data only.'),
          p(f'The eligible sample contains {O["eligible_stock_buckets"]:,} stock-bucket observations. Two buckets failed the coverage/positive-volatility criteria. I allocated complete time-ID groups to training ({O["split_rows"]["train"]:,} rows), validation ({O["split_rows"]["validation"]:,}) and test ({O["split_rows"]["test"]:,}). All stocks from a simultaneous bucket remain together.'),
          p('The bucket IDs are shuffled. This is a reserved group test, not a chronological out-of-time test. It prevents the same simultaneous bucket from appearing in both training and test data, but cannot establish future-calendar generalization.'),
          heading('Forecast comparison')]
    val=SCORES[SCORES.split=='validation'].set_index('model');test=SCORES[SCORES.split=='test'].set_index('model')
    rows=[['Forecast','Validation RMSPE','Test RMSPE']]
    for name in test.index:rows.append([name,percent(val.loc[name,'rmspe']),percent(test.loc[name,'rmspe'])])
    a+=table(rows,[240,138,138])
    a+=[p('Persistence repeats the first-half volatility. Calibrated persistence multiplies that forecast by a training-only factor chosen to minimize relative squared error. RV-only Ridge uses previous volatility and stock identity; the full Ridge adds market-data features. The two full models have fixed settings, and validation RMSPE selects Ridge over gradient boosting.'),PageBreak(),
        heading('2. How much value did the added data contribute?'),
        chart('optiver_model_comparison.png',width=500),Spacer(1,10),
        p(f'The full Ridge model reduced test RMSPE by {percent(O["relative_rmspe_improvement"],1)} relative to uncalibrated persistence. Against calibrated persistence, the improvement was {percent(O["relative_improvement_vs_calibrated_persistence"],1)}. Against RV-only Ridge, it was {percent(O["relative_improvement_vs_rv_only_ridge"],1)}.'),
        p(f'The 95% bucket-group bootstrap interval for improvement versus calibrated persistence runs from {percent(O["group_bootstrap_calibrated_improvement_95_ci"][0],1)} to {percent(O["group_bootstrap_calibrated_improvement_95_ci"][1],1)}. It includes no improvement. The corresponding interval versus RV-only Ridge is {percent(O["group_bootstrap_rv_only_improvement_95_ci"][0],1)} to {percent(O["group_bootstrap_rv_only_improvement_95_ci"][1],1)}.'),
        p('That changes the interpretation. Much of the improvement over the naive forecast can be obtained by calibration. The additional market-data features contribute a smaller gain in this sample. I would prioritize the simple model and stronger temporal validation before adding complexity.'),
        heading('What the forecast can support'),
        p('A short-horizon volatility estimate can inform how a trader evaluates changing risk, position size, or execution conditions. This study does not test those decisions directly. It forecasts price variability, not price direction, fill quality, or profit.'),
        p('The bootstrap keeps simultaneous stocks together but treats the anonymized buckets as exchangeable. Unknown calendar ordering prevents a serial-dependence-aware temporal audit. Three anonymized stocks also limit how broadly the result can be applied.','SmallCase'),PageBreak(),
        heading('3. Trading-strategy test'),
        p('I tested a fixed, long-only SPY rule. A daily close above its trailing 200-session price average generates a long signal; a close at or below the average generates a cash signal. Each decision executes at the following session\'s close. The new position first earns the next close-to-close return.'),
        p('The model charges 5 basis points on one-way traded notional, has no leverage, and assumes cash earns zero. These are explicit research assumptions. Returns use Yahoo adjusted closing prices as a dividend/split-adjusted total-return proxy. A separate daily-rebalanced 50% ETF/50% cash benchmark helps distinguish timing from simply holding less market exposure.'),
        p('Daily price history begins January 2, 2001. The simulation begins January 2, 2002 after the rolling-average warm-up. Development is 2002-2014, validation is 2015-2019, and the later historical evaluation is January 2, 2020 through September 30, 2026. The 200-session rule was fixed before reviewing results.'),
        chart('SPY_held_out_performance.png',width=475),Spacer(1,8)]
    a+=table([['Later historical period','Net CAGR','Volatility','Max. drawdown'],
              ['Trend filter',percent(T['cagr']),percent(T['annualized_volatility']),percent(T['max_drawdown'])],
              ['Buy and hold',percent(B['cagr']),percent(B['annualized_volatility']),percent(B['max_drawdown'])],
              ['50% ETF / 50% cash',percent(F['cagr']),percent(F['annualized_volatility']),percent(F['max_drawdown'])]], [210,94,102,110])
    a+=[p('CAGR and volatility use 252 trading sessions per year. Drawdown includes the period\'s starting value. Positions carry across research-period boundaries; all portfolios liquidate on the final date. All transactions shown are simulated.','SmallCase'),PageBreak(),
        heading('4. Robustness and decision implications'),
        p(f'The primary rule was invested for {percent(T["average_exposure"],1)} of the later period. It reduced SPY\'s annualized volatility from {percent(B["annualized_volatility"])} to {percent(T["annualized_volatility"])}, but its CAGR was {(B["cagr"]-T["cagr"])*100:.2f} percentage points lower. The simple 50% stock/cash portfolio had lower volatility and a smaller drawdown, with lower return.'),
        p('The historical regime matters. In 2002-2014, the trend filter produced a 6.86% CAGR versus 6.59% for buy-and-hold, with much lower maximum drawdown. In 2015-2019, its CAGR fell to 6.68% versus 11.58%. The stronger early result did not carry through consistently.'),
        heading('Transfer checks: the same rule and costs')]
    rows=[['ETF','Trend CAGR','Buy/hold CAGR','Trend drawdown','Buy/hold drawdown']]
    for ticker in ['SPY','QQQ','IWM']:
        f=PERF[(PERF.ticker==ticker)&(PERF.period=='Held-out')].set_index('strategy')
        tr=f.loc['Trend filter'];bh=f.loc['Buy and hold']
        rows.append([ticker,percent(tr.cagr),percent(bh.cagr),percent(tr.max_drawdown),percent(bh.max_drawdown)])
    a+=table(rows,[64,100,110,119,123])
    a+=[p('The same trade-off appears across all three ETFs: lower observed drawdown, but lower CAGR than buy-and-hold. This supports a risk-management interpretation rather than a broad claim of return outperformance.'),
        heading('Cost and rule sensitivity'),
        p('In the SPY later-period test, increasing one-way cost from 0 to 25 basis points reduced the primary rule\'s CAGR from 10.72% to 8.96%. At 5 basis points, the 50-, 100-, 200- and 250-session rules produced CAGRs of 8.48%, 12.00%, 10.36% and 8.45%. These are robustness comparisons; the best result was not substituted for the primary rule.'),
        p('An additional one-session execution delay produced 9.87% CAGR; an additional two-session delay produced 12.15%. A constant illustrative 3% cash yield produced 11.05%. The cash-yield test is not a reconstruction of historical Treasury or money-market returns.'),
        p('A 20-session moving-block bootstrap interval for annualized mean return difference versus buy-and-hold includes both positive and negative values. It does not establish a statistically reliable return advantage. Historical knowledge may also influence the choice of research question; this holdout is not prospective live testing.'),PageBreak(),
        heading('5. Research conclusion and verification'),
        p('I would retain the order-book study as evidence of a complete research process: check the input data, define information available at the decision point, compare simple and more complex forecasts, reserve evaluation groups, and explain the result against a meaningful baseline.'),
        p('I would retain the ETF test as evidence of investment judgment. A rule can make a portfolio less volatile and still leave the investor worse off on return. The decision depends on the purpose of the portfolio, tolerance for drawdown, the value of time in cash, and implementation costs.'),
        p('The next useful research step would be dated intraday data, temporal model validation, and a specific execution or position-sizing decision tied to the volatility forecast. This project stops at the evidence currently supported by its datasets.'),
        heading('Checks completed'),
        p('Source-file checksums and record counts agree. SQL checks found no duplicate feature keys, split leakage groups, invalid volatility targets, or duplicate ETF dates. Changing future quotes and trades did not change first-half features. Changing future ETF prices did not change earlier signals.'),
        p('A separate cash-and-fractional-holdings ledger agrees with the vectorized strategy accounting, including transaction costs and both benchmarks. An execution-timing fixture confirms that a new position cannot earn the return before its modeled entry. CAGR and drawdown were independently recomputed from the saved daily returns.'),
        heading('Sources and reproducibility'),
        p('<b>Optiver data:</b> <link href="https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data" color="#2563eb">Kaggle competition dataset</link>. Files retrieved from a <link href="https://github.com/KenChiang1997/Optiver-Realized-Volatility-Prediction/tree/c23a8f4feab7834106703d3e035a1fddc40234c4" color="#2563eb">fixed-commit public mirror</link>. The manifest records each original path and checksum. The mirror is a third-party copy, not a live Optiver data feed.','SmallCase'),
        p('<b>Bucket structure:</b> <link href="https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/249752" color="#2563eb">Optiver host dataset clarification</link>. Time IDs are shuffled; the original target uses the following ten minutes.','SmallCase'),
        p('<b>ETF prices:</b> Yahoo Finance chart API snapshots retrieved October 7, 2026. Raw JSON, corporate actions, and daily CSVs are retained. Dividend dates were checked; adjusted returns can differ slightly from a simple cash-dividend calculation because of adjustment conventions.','SmallCase'),
        p('<b>ETF context:</b> <link href="https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy" color="#2563eb">State Street SPY information</link>. <b>Methods:</b> <link href="https://scikit-learn.org/stable/modules/cross_validation.html" color="#2563eb">scikit-learn validation guidance</link>, <link href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html" color="#2563eb">Ridge documentation</link>, and <link href="https://mebfaber.com/2009/02/19/a-quantitative-approach-to-tactical-asset-allocation-updated/" color="#2563eb">Faber\'s moving-average research</link> for background. This implementation uses its own stated daily timing and cost assumptions.','SmallCase'),
        p('Reproduction: run the saved scripts against the cached snapshots. Downloading fresh prices can produce different adjusted-price histories. Source and model settings are recorded in research_plan.json; all principal outputs are in 04_Results.','SmallCase')]
    path=REPORTS/'Florien_Toukam_Market_Research_Memorandum.pdf'
    build(path,a,'Market Data Research and Trading Strategy Evaluation')
    return path

def study_guide():
    a=[p('Project Study Guide','TitleCase'),
       p('Understand the analysis, reproduce the results, and explain your decisions clearly.','SubtitleCase'),
       heading('1. Start with the two questions'),
       p('<b>Question one:</b> Does the information in an order book and trade history help estimate volatility over the next five minutes? <b>Question two:</b> Does a simple long/cash trend rule improve the trade-off between investment return and market risk?'),
       p('The first analysis uses Optiver records. The second uses dated ETF prices. They belong in one market-research case because both connect data to an investment decision, but they are separate experiments. The volatility model does not generate the ETF strategy\'s trades.'),
       heading('Read the project in this order'),
       p('First read the research memorandum. Then inspect optiver_model_scores.csv, strategy_performance.csv, and strategy_sensitivity.csv. Next open the research notebook and look at the feature-building and signal-timing code. Finally inspect a few rows from the simulated trade log.'),
       heading('The findings to remember'),
       p('Ridge forecast error was 28.67% RMSPE versus 34.22% for a naive repeat-volatility forecast. That is a 16.2% relative error reduction. Against a calibrated baseline, the reduction was only 3.4%, with uncertainty including no improvement. Against a previous-volatility/stock-ID Ridge model, the reduction was 4.3%.'),
       p('In the later SPY period, the trend rule produced 10.36% annualized growth versus 15.29% for buy-and-hold. Its maximum drawdown was 23.20% versus 33.72%. It reduced risk and also gave up return. The 50% stock/cash benchmark had an even smaller drawdown.'),
       p('This case was reconstructed on October 7, 2026. Study it as newly completed project work. Its figures do not belong inside an older employment accomplishment, and the simulated trades are not actual trading experience.','SmallCase'),PageBreak(),
       heading('2. Understand the market-data fields'),
       p('<b>Bid:</b> a displayed price at which a buyer is willing to buy. <b>Ask:</b> a displayed price at which a seller is willing to sell. <b>Spread:</b> ask minus bid. <b>Depth:</b> displayed shares available at the quoted price. <b>Trade:</b> an executed market transaction in the source dataset.'),
       p('The Optiver prices are normalized. Stock IDs do not identify real tickers. The share sizes and normalized price relationships support within-bucket analysis, but not a calendar-aware strategy built by sorting the shuffled time IDs.'),
       heading('The calculations used'),
       p('<b>Mid-price</b> = (bid + ask) / 2.<br/><b>Relative spread in basis points</b> = (ask - bid) / mid-price x 10,000.<br/><b>Imbalance</b> = (bid size - ask size) / (bid size + ask size).<br/><b>Weighted average price (WAP)</b> = (bid x ask size + ask x bid size) / total top-level size.'),
       p('For example, with bid 100.00, ask 100.04, bid size 800 and ask size 200, the mid-price is 100.02 and WAP is 100.032. The formula weights the opposite quote by displayed size. It describes current book conditions; it does not guarantee the next price move.'),
       heading('Realized volatility'),
       p('For successive WAP observations, compute log(current WAP / previous WAP). Square each log return, add them, and take the square root. That gives realized volatility for the observed window. Log returns are calculated separately inside each five-minute half and each stock/time bucket.'),
       p('The first five minutes provide the predictors. The second five minutes provide the value to predict. A future-information check changes second-half quotes, depth and trades and confirms that first-half predictors remain identical.'),
       heading('What the record count means'),
       p(f'There are {O["total_book_rows"]+O["total_trade_rows"]:,} underlying quote and trade records, but only {O["eligible_stock_buckets"]:,} aggregated modeling rows. Do not describe millions of ticks as millions of independent model training examples.'),PageBreak(),
       heading('3. Understand the model test'),
       p('<b>Training:</b> the model estimates relationships and learns scaling from this data. <b>Validation:</b> compares the two predefined full-model candidates. <b>Test:</b> the reserved groups used for the final reported comparison.'),
       p('All simultaneous observations sharing a time ID stay in one split. Otherwise, a model could train on stock 0 in a market interval and be tested on stock 1 from that same interval. This grouping addresses that overlap. It does not solve the unknown chronological order of different buckets.'),
       p('<b>Ridge regression</b> is linear regression with a penalty that discourages unstable coefficients. This model predicts log volatility, and exponentiation converts predictions back to positive volatility. Numeric features are standardized using training means and standard deviations.'),
       p('<b>Gradient boosting</b> combines small decision trees to capture nonlinear relationships. Here its settings were fixed, not searched over many alternatives. Ridge had lower validation RMSPE and remained the selected full model even after test results were examined.'),
       heading('Do not confuse these measurements'),
       p('<b>RMSPE</b> is the square root of average squared relative prediction error. If observed volatility is 0.20% and predicted volatility is 0.25%, the relative error is 25%. A 28.67% RMSPE is not a 28.67% investment return or 71.33% classification accuracy.'),
       p('<b>Relative error reduction</b> = (baseline error - model error) / baseline error. Using 34.22% and 28.67% gives about 16.2%. Using the stronger 29.69% calibrated baseline gives about 3.4%.'),
       p('A bootstrap repeatedly resamples whole test buckets to assess how much the comparison varies within the sample. The interval versus calibrated persistence crosses zero. That is a reason to be cautious about the small incremental gain. Unknown bucket chronology remains a limitation even when a resampling interval is positive.'),
       p('Read optiver_ridge_coefficients.csv as a description of the fitted model. Coefficients are conditional associations among correlated inputs, not evidence that changing one input causes volatility to move.','SmallCase'),PageBreak(),
       heading('4. Follow a signal through execution'),
       p('Suppose Monday closes above the trailing 200-session average. The model observes that completed close and plans to buy at Tuesday\'s close. It stays in its old position during Tuesday. The new position starts earning the return from Tuesday\'s close to Wednesday\'s close.'),
       p('This is why the return-position series is effectively delayed by two rows relative to the close signal. The implementation makes the order of observation, execution and return exposure explicit. The Tuesday closing signal can change again before the previously scheduled order executes.'),
       heading('A real historical example from the saved simulation')]
    prices=pd.read_csv(ROOT/'01_Data/raw/etf/SPY_daily_prices.csv',parse_dates=['date']).set_index('date')
    ledger=pd.read_csv(ROOT/'04_Results/SPY_trend_filter_daily_ledger.csv',parse_dates=['date']).set_index('date')
    subset=ledger.loc['2020-03-02':'2020-03-05']
    rows=[['Date','Close signal','Weight during day','Weight after close']]
    for date,row in subset.iterrows():rows.append([str(date.date()),'LONG' if row.signal_at_today_close else 'CASH',f'{row.weight_held_during_session:.0%}',f'{row.weight_after_close:.0%}'])
    a+=table(rows,[120,118,139,139])
    a+=[p('A scheduled buy can occur on a day whose new closing signal is CASH. That is a consequence of the execution delay, not evidence of a mislabeled signal. Rapid reversals near the moving average produce whipsaws, turnover and costs.'),
        heading('Costs and cash'),
        p('Five basis points is 0.05% of one-way traded notional. A modeled $100,000 purchase therefore costs roughly $50, with the exact invested amount adjusted for the fee. Selling incurs another one-way cost. The return ledger deducts these fees and recomputes holdings.'),
        p('Cash earns zero in the main test. The 3% cash-yield sensitivity is an illustrative constant. Fees shown in dollars use the continuing account started with $100,000 in 2002; they are not automatically rebased to a new $100,000 account in 2020.'),
        p('Adjusted prices represent a total-return proxy with distributions reflected in adjustments. The independent ledger holds fractional proxy units. It is an accounting cross-check, not a literal broker statement containing SPY share quantities and cash dividend payments.'),PageBreak(),
        heading('5. Explain the risk-return trade-off'),
        p('<b>Total return:</b> ending value / starting value - 1. <b>CAGR:</b> the compound growth rate, annualized using 252 daily observations per year. <b>Annualized volatility:</b> daily return standard deviation x square root of 252. <b>Maximum drawdown:</b> the largest percentage fall from the previous portfolio high, including the period\'s starting value.'),
        p('The reported Sharpe-style ratio uses a zero-return hurdle. It is not calculated using a historical Treasury risk-free series. Avoid presenting it as a full risk-adjusted alpha estimate.'),
        p('Average exposure is the share of sessions invested for the binary trend rule. SPY exposure was 79.0% in the later period. Some risk reduction naturally comes from spending time in cash. The 50% stock/cash benchmark makes that trade-off visible.'),
        heading('Why the testing periods matter'),
        p('Development: 2002-2014. Validation: 2015-2019. Later historical evaluation: 2020-September 2026. The rule is fixed across these periods. Performance improves or worsens with the market regime; a good early result does not guarantee a good later one.'),
        p('The sensitivity file contains alternate lookbacks, costs, execution delays and cash assumptions. It is used to understand fragility. Picking the best later-period variant and then reporting it as though it had been the original rule would overstate the evidence.'),
        heading('How to describe the project clearly'),
        p('A useful explanation is: "I built two market-data tests. I used order-book and trade features to forecast short-horizon volatility, then compared a delayed trend strategy with passive benchmarks. The forecast improved on a naive baseline, but stronger baselines narrowed the gain. The strategy reduced drawdown and volatility while giving up return, so I treated it as a risk-management trade-off."'),
        p('Be ready to explain why the Optiver split is grouped rather than chronological, why the strategy executes one session later, why calibration matters, and why predicting volatility does not tell you whether a price will rise or fall.'),PageBreak(),
        heading('6. Reproduce and inspect the work'),
        p('The complete raw data is in 01_Data/raw. Aggregated features and the SQLite database are in 01_Data/processed. The notebook is in 02_Notebooks. The analysis scripts and pinned package versions are in 03_Code. Saved tables and verification files are in 04_Results.'),
        heading('What each tool does'),
        p('<b>Python/Pandas:</b> reads parquet and price data, validates keys, groups market records, builds predictors and aligns signals. <b>NumPy:</b> calculates log returns, volatility, costs, portfolio returns and bootstrap samples. <b>Scikit-learn:</b> fits training-only preprocessing and the forecast models. <b>SQL/SQLite:</b> independently checks processed-table keys, split membership and price-date uniqueness.'),
        heading('One-run reproduction'),
        p('On this computer, RUN_ANALYSIS.command uses the project\'s local Python environment and the cached data. It rebuilds the analysis, verification files, reports and notebook. It does not download data again. To refresh source data deliberately, use download_data.py, then rerun the analysis; revised vendor adjustments may change the results.'),
        heading('Questions to answer before using the project on your resume'),
        p('1. What was the prediction target and which data was available before it?<br/>2. What does a time ID mean, and why is sorting it not a valid backtest?<br/>3. What happened to the model\'s advantage when the baseline was calibrated?<br/>4. When does a signal become a position that earns a return?<br/>5. How did trading costs and cash exposure affect the strategy?<br/>6. Why was a 50% stock/cash benchmark useful?<br/>7. What would you need to test before using either result in an actual trading process?'),
        heading('The practical conclusion'),
        p('This case gives you concrete research to discuss: market-data preparation, a forecast comparison, execution timing, transaction costs, portfolio risk and a written investment conclusion. The value comes from explaining the decisions and limits of the analysis, including the findings that did not support outperformance.'),
        p('For formulas, split definitions, settings and source links, refer to the memorandum and research_plan.json. The notebook contains saved-result explanations and code cells for rerunning the analysis; it is not a record of past professional work.','SmallCase')]
    path=REPORTS/'Florien_Toukam_Market_Research_Study_Guide.pdf'
    build(path,a,'Market Research Project Study Guide')
    return path

def main():
    paths=[memorandum(),study_guide()]
    for path in paths:
        r=PdfReader(path)
        print(path.name,len(r.pages),'pages',path.stat().st_size,'bytes',flush=True)
        assert len(r.pages)==6, 'Unexpected page overflow: inspect layout before delivery'

if __name__=='__main__':main()
