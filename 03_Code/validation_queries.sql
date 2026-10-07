-- Processed research tables, not professional client accounts.
-- Duplicate feature keys: must be zero.
SELECT COUNT(*) AS duplicate_feature_keys
FROM (SELECT stock_id, time_id FROM optiver_features GROUP BY stock_id, time_id HAVING COUNT(*) > 1);

-- The same simultaneous market bucket cannot appear in multiple splits: must be zero.
SELECT COUNT(*) AS split_leakage_groups
FROM (SELECT time_id FROM optiver_features GROUP BY time_id HAVING COUNT(DISTINCT split) > 1);

-- Target and previous volatility must be defined and positive: must be zero.
SELECT COUNT(*) AS invalid_volatility_rows FROM optiver_features
WHERE rv_first IS NULL OR rv_target IS NULL OR rv_first <= 0 OR rv_target <= 0;

-- Confirm the sample and split dimensions independently.
SELECT stock_id, split, COUNT(*) AS stock_buckets, AVG(spread_bps) AS mean_quote_spread_bps
FROM optiver_features GROUP BY stock_id, split ORDER BY stock_id, split;

-- ETF price uniqueness: must be zero.
SELECT COUNT(*) AS duplicate_etf_dates
FROM (SELECT ticker, date FROM etf_prices GROUP BY ticker, date HAVING COUNT(*) > 1);

-- Inspect actual simulated trading events, including the delayed close signal.
SELECT date, action, weight_held_during_session, weight_after_close,
       close_signal_price, signal_at_today_close, turnover_fraction, fee_dollars
FROM spy_trade_log WHERE date BETWEEN '2020-02-01' AND '2020-06-30' ORDER BY date;
