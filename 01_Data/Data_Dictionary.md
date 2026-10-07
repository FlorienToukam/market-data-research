# Data dictionary

## Raw Optiver files

- `time_id`: shuffled identifier joining simultaneous market buckets across stocks. Not a timestamp or sortable calendar sequence.
- `seconds_in_bucket`: 0-599 second offset in the observed ten-minute interval.
- `bid_price1`, `ask_price1`: normalized best bid and ask.
- `bid_price2`, `ask_price2`: normalized second-level prices.
- `bid_size1/2`, `ask_size1/2`: displayed share sizes at each level.
- `price`: normalized price of executed trade records.
- `size`: trade volume in the aggregated trade record.
- `order_count`: number of executions within the trade record.
- Stock ID is provided by the source partition, not a ticker.

## Processed Optiver features

- `rv_first`: first-five-minute realized volatility using WAP log changes.
- `rv_target`: second-five-minute realized volatility. Never a model predictor.
- `spread_bps`: mean first-half relative top-level spread; one basis point is 0.01%.
- `depth`: mean first-half top-level displayed size.
- `imbalance`: mean first-half signed size imbalance, between -1 and 1.
- `abs_imbalance`: mean absolute first-half imbalance.
- `quote_observations`: number of first-half quote observations.
- `trade_volume`, `trade_records`, `execution_count`: first-half trade totals and counts.
- Coverage columns record the earliest/latest second and observation counts for each half.
- `log_*`: model transformations; log volatility, log(1+x) for nonnegative size/count/spread fields.
- `split`: train, validation or test, assigned by the complete time-ID group.

Eligibility requires at least ten quotes in each half, first-half coverage reaching from second <=60 to >=240, second-half coverage reaching from <=360 to >=540, and positive volatility in both halves. Two of 11,490 stock buckets are excluded: stock 0 time IDs 13484 and 22079 end at seconds 513 and 503, respectively, short of the required second-half coverage. Their rows are saved in `04_Results/optiver_excluded_buckets_stock_0.csv`. No target value is used to choose model features or tune parameters; positive target eligibility is necessary for the relative-error metric.

## ETF prices

- `date`: New York exchange-local calendar date.
- `open/high/low/close`: Yahoo price fields; Close is the split-adjusted price series excluding dividend reinvestment.
- `adjusted_close`: vendor-adjusted dividend/split return proxy used to calculate performance.
- `volume`: source trading volume.
- Corporate action JSONs retain dividend dates/amounts and supplied split events.

The data source's adjusted-price return convention can differ slightly from adding the cash dividend to the closing price. Non-dividend dates and all dividend dates are audited in `etf_data_quality.csv`.

## Daily portfolio ledger

- `asset_return`: daily adjusted-close return.
- `weight_held_during_session`: allocation earning today's market return.
- `weight_after_close`: allocation after the modeled close execution; final date is liquidated.
- `pre_trade_weight`: allocation after today's price movement but before rebalancing.
- `turnover_fraction`: actual one-way modeled traded notional / pre-trade account value.
- `fee_fraction`, `fee_dollars`: modeled cost paid on one-way traded notional.
- `gross_return`: pre-trading-cost portfolio return.
- `net_return`: post-cost portfolio return.
- `wealth`: continuous account value from $100,000 at the 2002 simulation start.
- `close_signal_price`, `trailing_sma`, `signal_at_today_close`: audit fields for the new end-of-day decision.
- `adjusted_close_return_proxy`: price used for performance accounting, not the raw quoted execution price.

The independent cash/units ledger uses units of this return proxy to check accounting. The modeled positions are not literal broker share holdings. Period performance is computed from daily returns with positions carried across boundaries. Dollar fees use the continuing 2002 account; they are not rebased to $100,000 in each subperiod.

## Metrics

RMSPE = sqrt(mean(((prediction - observed) / observed)^2)). MAE/RMSE are in volatility units. A raw volatility value of 0.001 is 0.1%, or 10 basis points.

CAGR = product(1 + daily net returns)^(252 / number of observations) - 1. Volatility = daily sample standard deviation x sqrt(252). The reported Sharpe-style ratio uses a zero hurdle, not historical Treasury returns. Drawdown compares the cumulative portfolio value with its previous peak including the starting value.
