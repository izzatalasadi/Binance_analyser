# Binance_analyser
This project is an asynchronous cryptocurrency analysis bot that integrates with the Binance API and performs real-time data analysis. The bot uses various technical indicators to make trading decisions, and to handle API rate limits and automatically reconnect to WebSocket streams when disconnected.

## Parameters
- **api_key**: Binance API key.
- **api_secret**: Binance API secret key.
- **max_retries (optional)**: Maximum number of retries for failed requests. Defaults to 5.
- **max_concurrent_requests (optional)**: Maximum number of concurrent API requests. Defaults to 5.


## Features
- **Market Data**: Fetches real-time and historical market data from Binance.
- **Technical Indicators**: Calculates RSI, MACD, VWAP, Bollinger Bands, ATR, and more.
- **News Sentiment**: Analyzes cryptocurrency news sentiment using the `CryptoPanicScraper`.
- **Backtesting**: Backtests the trading strategy on historical data.
- **Order Book**: Fetches and analyzes the order book for real-time predictions.
- **WebSocket Streaming**: Supports real-time data streaming through Binance WebSocket.

## Methods
- **initialize_client(self)**: Initializes the Binance AsyncClient and the WebSocket manager.
- **close_client(self)**: Closes the Binance AsyncClient session.
- **rate_limited_request(self, func, *args, **kwargs)**: Makes API requests with retry logic for rate limits and backoff handling.
- **get_ticker_price(self, ticker_symbol, days, granularity)**: Fetches historical kline data for a given ticker symbol and granularity.
- **stream_ticker(self, ticker_symbol)**: Streams real-time price data for the given ticker symbol using WebSocket.
- **get_price_data(self, tickers, days, granularity)**: Fetches price data for multiple tickers asynchronously.
- **generate_report(self, ticker, indicators)**: Generates a detailed report of the technical indicators for a given ticker symbol.
- **get_order_book(self, ticker_symbol)**: Fetches the order book for a given ticker symbol asynchronously.
- **predict_price(self, rsi, bids, asks, short_ma, long_ma, vwap, macd, macd_signal, bb_upper, bb_lower, atr, stochastic, obv, weights)**: Predicts the price and action (buy/sell/hold) based on technical indicators and order book data.
- **pick_coins(self, cointickers, day_corr, week_corr, two_week_corr, size_of_list)**: Selects the top-performing coins based on correlations with close prices.
- **backtest_strategy(self, historical_data, weights, transaction_cost, slippage)**: Backtests the trading strategy on historical data.
- **get_news_and_sentiment(self, ticker, bridge, filter, news_limit)**: Fetches news and analyzes sentiment for a given ticker using CryptoPanicScraper.
- **get_top_gainers(self)**: Fetches the top gainers (coins with the highest percentage price change).
- **get_top_losers(self)**: Fetches the top losers (coins with the lowest percentage price change).
- **get_market_analysis(self, symbol)**: Fetches and calculates market analysis (moving averages, RSI) for the specified symbol.
- **get_market_performance(self, symbol)**: Fetches the latest market performance for the specified symbol.
- **get_historical_data(self, symbol, days, granularity)**: Fetches historical price data for a specific symbol.
- **get_recommendation(self, symbol)**: Provides a buy/sell/hold recommendation based on technical analysis and historical data.
- **run(self)**: Main method for running the bot, collecting data, performing analysis, and making predictions.

  ## Requirements
- `python` (>=3.8)
- `binance`
- `asyncio`
- `ssl`
- `nltk`
- `certifi`
- `pandas`
- `numpy`

### Install required packages:
pip install binance asyncio ssl pandas numpy certifi
