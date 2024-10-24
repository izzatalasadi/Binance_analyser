import logging
import asyncio
import ssl
import traceback
from binance import BinanceSocketManager
from binance.client import AsyncClient  
from binance.exceptions import BinanceAPIException, BinanceOrderException
from datetime import datetime, timedelta
import certifi
import pandas as pd
import numpy as np
from logging_config import logger

ssl_context = ssl.create_default_context(cafile=certifi.where())

class TradingBot:
    def __init__(self, api_key, api_secret, max_retries=5, max_concurrent_requests=5):
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.socket_manager = None  # Will be initialized for WebSocket streaming
        logger.info("Trading Bot initialized")

    async def initialize_client(self):
        # Initialize Binance AsyncClient with the custom session
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        self.socket_manager = BinanceSocketManager(self.client)
        
        logger.info("Binance AsyncClient initialized")

    async def close_client(self):
        """Close the Binance AsyncClient session."""
        await self.client.close_connection()
        logger.info("Binance AsyncClient connection closed")

    async def rate_limited_request(self, func, *args, **kwargs):
        retries = 0
        max_wait_time = 60  # Maximum wait time in seconds before giving up
        while retries < self.max_retries:
            try:
                return await func(*args, **kwargs)
            except BinanceAPIException as e:
                if e.status_code == 429:  # Too many requests
                    retries += 1
                    wait_time = min(2 ** retries, max_wait_time)  # Exponential backoff with cap
                    jitter = np.random.uniform(0, 1)  # Add randomness to reduce contention
                    wait_time += jitter
                    logger.error(f"Rate limit hit. Retrying in {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def get_ticker_price(self, ticker_symbol: str, days: int, granularity: str) -> pd.DataFrame:
        """
        Fetch historical kline data for a given ticker symbol asynchronously.
        """
        target_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        retries = 0

        while retries < self.max_retries:
            async with self.semaphore:
                try:
                    historical_data = await self.client.get_historical_klines(
                        symbol=ticker_symbol,
                        interval=granularity,
                        start_str=target_date.strftime("%d %b %Y %H:%M:%S"),
                        end_str=end_date.strftime("%d %b %Y %H:%M:%S"),
                        limit=1000
                    )
                    # Convert the data into a DataFrame
                    df = pd.DataFrame(
                        historical_data,
                        columns=[
                            'Open time', 'Open', 'High', 'Low', 'Close',
                            'Volume', 'Close time', 'Quote asset volume',
                            'Number of trades', 'Taker buy base asset volume',
                            'Taker buy quote asset volume', 'Ignore'
                        ]
                    )
                    df['date'] = pd.to_datetime(df['Open time'], unit='ms', utc=True)
                    df[ticker_symbol] = df['Close'].astype(float)

                    # Check for NaN values
                    if df.isnull().values.any():
                        logger.warning(f"Fetched data for {ticker_symbol} contains NaN values.")
                        df.dropna(inplace=True)

                    return df[['date', ticker_symbol, 'High', 'Low', 'Close', 'Volume']]

                except BinanceAPIException as e:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.error(f"Error fetching data for {ticker_symbol}: {e}. Retry {retries}/{self.max_retries}. Waiting {wait_time} seconds.", exc_info=True)
                    await asyncio.sleep(wait_time)
                except BinanceOrderException as e:
                    logger.error(f"Binance Order Exception for {ticker_symbol}: {e}")
                    break
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.error(f"General Exception for {ticker_symbol}: {e}. Retry in {wait_time} seconds.")
                    await asyncio.sleep(wait_time)

        logger.error(f"Failed to fetch data for {ticker_symbol} after {self.max_retries} retries.")
        return pd.DataFrame()

    async def stream_ticker(self, ticker_symbol):
        try:
            async with self.socket_manager.trade_socket(ticker_symbol) as stream:
                while True:
                    res = await stream.recv()
                    logger.info(f"Real-time data for {ticker_symbol}: {res}")
        except Exception as e:
            logger.error(f"Error in WebSocket stream for {ticker_symbol}: {e}")
            await self.reconnect_stream(ticker_symbol)

    async def reconnect_stream(self, ticker_symbol):
        logger.info(f"Reconnecting WebSocket stream for {ticker_symbol}")
        await asyncio.sleep(5)  # Small delay before reconnecting
        await self.stream_ticker(ticker_symbol)

    async def get_price_data(self, tickers, days=1, granularity="1m") -> pd.DataFrame:
        """
        Collect price and volume data for multiple tickers asynchronously.
        """
        tasks = [self.get_ticker_price(tick, days, granularity) for tick in tickers]
        results = await asyncio.gather(*tasks)

        coindata = pd.DataFrame()
        for res in results:
            if not res.empty:
                ticker = [col for col in res.columns if col not in ['date', 'High', 'Low', 'Close', 'Volume']][0]
                res = res.rename(columns={
                    ticker: f"Price_{ticker}",
                    'High': f"High_{ticker}",
                    'Low': f"Low_{ticker}",
                    'Close': f"Close_{ticker}",
                    'Volume': f"Volume_{ticker}"
                })
                if coindata.empty:
                    coindata = res
                else:
                    coindata = coindata.merge(res, on='date', how='outer')

        coindata.sort_values('date', inplace=True)
        coindata.reset_index(drop=True, inplace=True)
        coindata.dropna(inplace=True)

        failures = [tick for i, tick in enumerate(tickers) if results[i].empty]
        if failures:
            logger.warning(f"The following coins do not have historical data: {failures}")

        return coindata
    
    def generate_report(self, ticker: str, indicators: dict):
        """
        Generate and log a detailed report for the given ticker.
        
        INPUT:
        ticker          : str   : The ticker symbol of the cryptocurrency
        indicators      : dict  : The technical indicators
        fundamental_data: dict  : The fundamental data (optional)
        """
        report = f"\n--- Analysis Report for {ticker} ---\n"
        # Technical Indicators
        report += f"RSI: {indicators['rsi'].iloc[-1]:.2f}\n"
        report += f"MACD: {indicators['macd'].iloc[-1]:.4f}, Signal Line: {indicators['macd_signal'].iloc[-1]:.4f}\n"
        report += f"ATR: {indicators['atr'].iloc[-1]:.4f}\n"
        report += f"Stochastic Oscillator: {indicators['stochastic'].iloc[-1]:.2f}\n"
        report += f"OBV: {indicators['obv'].iloc[-1]:.2f}\n"
        report += f"Bollinger Bands: Upper {indicators['bb_upper'].iloc[-1]:.4f}, Lower {indicators['bb_lower'].iloc[-1]:.4f}\n"
            
        logger.info(report)
    
    async def get_order_book(self, ticker_symbol: str):
        """
        Retrieves the order book for a given ticker symbol asynchronously.
        """
        retries = 0
        while retries < self.max_retries:
            async with self.semaphore:
                try:
                    order_book = await self.client.get_order_book(symbol=ticker_symbol)
                    bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity']).astype(float)
                    asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity']).astype(float)
                    return bids, asks
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.error(f"Error fetching order book for {ticker_symbol}: {e}. Retry {retries}/{self.max_retries}. Waiting {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
        logger.error(f"Failed to fetch order book for {ticker_symbol} after {self.max_retries} retries.")
        return pd.DataFrame(), pd.DataFrame()

    def predict_price(self, rsi, bids, asks, short_ma, long_ma, vwap, macd, macd_signal, bb_upper, bb_lower, atr, stochastic, obv, weights):
        """
        Enhanced price prediction model using multiple indicators, including news sentiment.
        """

        indicators = {
            'rsi': rsi,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'vwap': vwap,
            'macd': macd,
            'macd_signal': macd_signal,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'atr': atr,
            'stochastic': stochastic,
            'obv': obv
        }

        for name, series in indicators.items():
            if series.empty:
                logger.warning(f"{name} Series is empty. Cannot proceed with prediction.")
                return None, 'Hold'

            if len(series) < 1:
                logger.warning(f"{name} Series has insufficient data. Cannot proceed with prediction.")
                return None, 'Hold'

        # Access last values, with safety checks
        def safe_iloc(series, name):
            if isinstance(series, pd.Series) and not series.empty:
                try:
                    return series.iloc[-1]
                except (IndexError, KeyError, TypeError) as e:
                    logger.error(f"Series {name} has no valid data or could not retrieve the last element: {e}")
            else:
                logger.error(f"Series {name} is invalid or empty.")
            return None

        # Safely retrieve the last element of each indicator
        last_rsi = safe_iloc(rsi, 'rsi')
        last_short_ma = safe_iloc(short_ma, 'short_ma')
        last_long_ma = safe_iloc(long_ma, 'long_ma')
        last_vwap = safe_iloc(vwap, 'vwap')
        last_macd = safe_iloc(macd, 'macd')
        last_macd_signal = safe_iloc(macd_signal, 'macd_signal')
        last_bb_upper = safe_iloc(bb_upper, 'bb_upper')
        last_bb_lower = safe_iloc(bb_lower, 'bb_lower')
        last_atr = safe_iloc(atr, 'atr')
        last_stochastic = safe_iloc(stochastic, 'stochastic')
        last_obv = safe_iloc(obv, 'obv')

        # Check if any of the indicators could not be retrieved, return 'Hold'
        if any(val is None for val in [last_rsi, last_short_ma, last_long_ma, last_vwap, last_macd, last_macd_signal, last_bb_upper, last_bb_lower, last_atr, last_stochastic, last_obv]):
            logger.error("Some indicators returned None, recommending Hold action.")
            return None, 'Hold'

        # Calculate mid-price from order book
        if bids.empty or asks.empty:
            logger.warning("Order book is empty. Cannot compute mid-price.")
            return None, 'Hold'

        mid_price = (asks['price'].min() + bids['price'].max()) / 2

        # Use the weighted model for decision-making
        buy_pressure = 0
        sell_pressure = 0

        # Apply trading rules based on indicators
        if last_rsi < 25:
            buy_pressure += weights.get('rsi', 0)
        elif last_rsi > 75:
            sell_pressure += weights.get('rsi', 0)

        if last_short_ma > last_long_ma:
            buy_pressure += weights.get('ma', 0)
        else:
            sell_pressure += weights.get('ma', 0)

        if mid_price < last_vwap:
            buy_pressure += weights.get('vwap', 0)
        else:
            sell_pressure += weights.get('vwap', 0)

        if last_macd > last_macd_signal:
            buy_pressure += weights.get('macd', 0)
        else:
            sell_pressure += weights.get('macd', 0)

        if mid_price < last_bb_lower:
            buy_pressure += weights.get('bbands', 0)
        elif mid_price > last_bb_upper:
            sell_pressure += weights.get('bbands', 0)

        # Incorporate news sentiment
        buy_pressure += weights.get('news_sentiment', 0) 

        # Final decision
        if buy_pressure > sell_pressure:
            recommended_price = asks['price'].min()  # Buy
            action = 'Buy'
        elif sell_pressure > buy_pressure:
            recommended_price = bids['price'].max()  # Sell
            action = 'Sell'
        else:
            recommended_price = None
            action = 'Hold'

        return recommended_price, action

    def pick_coins(self, cointickers, day_corr, week_corr, two_week_corr, size_of_list, ticker_prefix='Close_'):
        '''
        Picks the coin that jointly maximizes the correlation for the whole coin list according to close price ("Close_").
        
        INPUT:
        cointickers  : LIST : List of coin tickers (e.g., ['BTCUSDT', 'ETHUSDT'])
        day_corr     : PD.CORR : Daily correlation data
        week_corr    : PD.CORR : Weekly correlation data
        two_week_corr: PD.CORR : Bi-weekly correlation data
        size_of_list : INT : Number of coins to select (default: 5)
        ticker_prefix: STR : Prefix for the close price column (default: 'Close_')
        
        RETURNS:
        coinlist : LIST : List of selected coins that maximize correlation based on close prices
        '''
        
        # Log and ensure size_of_list is an integer
        #logging.info(f"Initial size_of_list type: {type(size_of_list)}, value: {size_of_list}")
        try:
            size_of_list = int(size_of_list)
        except ValueError:
            logging.error(f"Invalid value for size_of_list: {size_of_list}")
            return []

        #logging.info(f"After conversion, size_of_list type: {type(size_of_list)}, value: {size_of_list}")

        # Initialize the coinlist with Close_ prefix
        coinlist = [ticker_prefix + ticker for ticker in cointickers]

        
        corrsum = pd.Series(dtype=float)  # Initialize as an empty Series for storing sum of correlations

        # Sum the correlations for each coin in the coinlist
        for i in range(size_of_list):
            #logging.info(f"Entering iteration {i} of the loop with size_of_list={size_of_list}")
            
            
            for coin in coinlist:
                # Filter out only the 'Close_' correlations
                day_corr_values = day_corr.get(coin, pd.Series(dtype='float64')).fillna(0)
                week_corr_values = week_corr.get(coin, pd.Series(dtype='float64')).fillna(0)
                two_week_corr_values = two_week_corr.get(coin, pd.Series(dtype='float64')).fillna(0)
                
                # Ensure we are only summing 'Close_' columns
                if corrsum.empty:
                    corrsum = day_corr_values + week_corr_values + two_week_corr_values
                else:
                    corrsum += day_corr_values + week_corr_values + two_week_corr_values

            # Log the current state of corrsum
            #logging.info(f"Current correlation sum: \n{corrsum}")

            # Handle cases where corrsum is empty or all NaN
            if corrsum.dropna().empty:
                logging.error("Correlation sum is empty or NaN. Stopping selection.")
                break

            # Remove negative correlations
            corrsum = corrsum[corrsum >= 0]

            # Find the next coin with the highest correlation sum
            try:
                ind = corrsum.dropna().idxmax()  # idxmax returns the index label
                next_coin = ind  # Now `ind` is already the label (e.g., 'Close_BTCUSDT')

                # Check if the next_coin is already in the coinlist
                if next_coin not in coinlist:
                    coinlist.append(next_coin)  # Add next_coin unconditionally if not already in list
                    
            except ValueError as e:
                logging.error(f"Error finding idxmax: {e}. Correlation sum is empty or invalid.")
                break

        # Strip the 'Close_' or any prefix from the final coinlist
        final_coinlist = [coin.replace(ticker_prefix, '') for coin in coinlist]

        # Return only the first n coins
        return final_coinlist[:size_of_list]

    def backtest_strategy(self, historical_data: pd.DataFrame, weights: dict, transaction_cost: float = 0.001, slippage: float = 0.0005):
        """
        Backtest the prediction model on historical data.
        """
        total_trades = 0
        successful_trades = 0
        total_profit = 0
        positions = []
        trade_log = []

        for index in range(50, len(historical_data) - 1):
            # Extract data slice
            data_slice = historical_data.iloc[:index]
            current_price = historical_data['Price'].iloc[index]

            # Calculate indicators
            price_series = data_slice['Price']
            volume_series = data_slice['Volume']
            high_series = data_slice['High']
            low_series = data_slice['Low']
            close_series = data_slice['Close']

            rsi = self.calculate_rsi(price_series)
            short_ma, long_ma = self.calculate_moving_averages(price_series)
            vwap = self.calculate_vwap(price_series, volume_series)
            macd, macd_signal = self.calculate_macd(price_series)
            bb_upper, bb_lower = self.calculate_bollinger_bands(price_series)
            atr = self.calculate_atr(high_series, low_series, close_series)
            stochastic = self.calculate_stochastic_oscillator(high_series, low_series, close_series)
            obv = self.calculate_obv(close_series, volume_series)

            # Mock order book
            bid_price = current_price * (1 - slippage)
            ask_price = current_price * (1 + slippage)
            bids = pd.DataFrame({'price': [bid_price], 'quantity': [1]})
            asks = pd.DataFrame({'price': [ask_price], 'quantity': [1]})

            # Prediction
            recommended_price, action = self.predict_price(
                rsi=rsi,
                bids=bids,
                asks=asks,
                short_ma=short_ma,
                long_ma=long_ma,
                vwap=vwap,
                macd=macd,
                macd_signal=macd_signal,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                atr=atr,
                stochastic=stochastic,
                obv=obv,
                weights=weights
            )

            if action is None or recommended_price is None or action == 'Hold':
                continue  # Skip to the next iteration

            total_trades += 1
            # Simulate trade
            if action == 'Buy':
                entry_price = recommended_price * (1 + transaction_cost)
                exit_price = historical_data['Close'].iloc[index + 1]  # Next period's close price
                profit = exit_price - entry_price
            elif action == 'Sell':
                entry_price = recommended_price * (1 - transaction_cost)
                exit_price = historical_data['Close'].iloc[index + 1]
                profit = entry_price - exit_price
            else:
                continue  # Skip if action is not 'Buy' or 'Sell'

            total_profit += profit
            if profit > 0:
                successful_trades += 1

            # Log the trade
            trade_log.append({
                'Date': historical_data['date'].iloc[index],
                'Action': action,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Profit': profit
            })

        # Convert trade log to DataFrame
        trade_log_df = pd.DataFrame(trade_log)

        # Calculate performance metrics
        accuracy = float((successful_trades / total_trades) * 100 if total_trades > 0 else 0)
        avg_profit = float(total_profit / total_trades if total_trades > 0 else 0)

        # Calculate Sharpe Ratio
        returns = trade_log_df['Profit']
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns))
        else:
            sharpe_ratio = 0

        # Calculate Maximum Drawdown
        cumulative_profit = trade_log_df['Profit'].cumsum()
        peak = cumulative_profit.expanding(min_periods=1).max()
        drawdown = (cumulative_profit - peak)
        max_drawdown = drawdown.min()

        # logging.info(
        # f"\nBacktest Results for {historical_data['symbol'].iloc[0]}:\n"
        # f"Trades: {total_trades}\nAccuracy: {accuracy:.2f}%\nAverage Profit per Trade: {avg_profit:.4f}\n"
        # f"Sharpe Ratio: {sharpe_ratio:.2f}\nMaximum Drawdown: {max_drawdown:.2f}")

        return accuracy, avg_profit

    async def get_trade_volume(self, coinlist):
        """
        Fetches and logs the 24-hour trade volume for the coins in the coinlist asynchronously.
        """
        try:
            tickers = await self.client.get_ticker()
            
            if not tickers:
                logger.warning("No ticker data returned from Binance API.")
                return
            volume_dict = {}

            for data in tickers:
                if data['symbol'] in coinlist:
                    usd_trade_volume = float(data['quoteVolume'])  # Use quoteVolume for USD volume
                    volume_dict[data['symbol']] = usd_trade_volume

            volume_df = pd.DataFrame(list(volume_dict.items()), columns=['Coin', 'Volume'])
            sorted_volume_df = volume_df.sort_values(by='Volume', ascending=False)

            for _, row in sorted_volume_df.iterrows():
                if row['Volume'] > 5e6:
                    logging.info(f"{row['Coin']} - 24hr trade volume: {row['Volume']:.0f} USD")
        except Exception as e:
            logging.error(f"Error fetching trade volumes: {e}")

    def has_sufficient_data(self, price_data: pd.Series,volume_data: pd.Series = None, min_required_points: int = 50) -> bool:
        """
        Checks if there is sufficient data for indicator calculations.
        """
        # Include all window sizes used in indicators
        if len(price_data) < min_required_points:
            logging.warning(
                f"Not enough data points. Only {len(price_data)} "
                f"available, {min_required_points} needed."
            )
            return False
        if len(price_data) < min_required_points:
            logging.warning(
                f"Not enough data points. Only {len(price_data)} available, {min_required_points} needed."
            )
            return False
        if price_data.isnull().any():
            logging.warning("Price data contains NaN values.")
            return False
        if volume_data is not None and (
            len(volume_data) < min_required_points or volume_data.isnull().any()
        ):
            logging.warning(
                "Volume data contains NaN values or insufficient data."
            )
            return False
        return True
    
    @staticmethod
    def calculate_obv(close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        """
        # Ensure no NaN values in the close or volume series
        close_series = close_series.ffill().fillna(0)
        volume_series = volume_series.ffill().fillna(0)
        
        if close_series.empty or volume_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        direction = close_series.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume_series * direction).cumsum()
        
        return obv

    @staticmethod
    def calculate_stochastic_oscillator(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate the Stochastic Oscillator (%K).
        """
        # Ensure no NaN values
        high_series = high_series.ffill().fillna(0)
        low_series = low_series.ffill().fillna(0)
        close_series = close_series.ffill().fillna(0)
        if close_series.empty or high_series.empty or low_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        lowest_low = low_series.rolling(window=window).min()
        highest_high = high_series.rolling(window=window).max()
        
        # Avoid division by zero by replacing invalid values with NaN
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)  # Avoid division by zero
        stochastic_k = 100 * ((close_series - lowest_low) / denominator)
        stochastic_k.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return stochastic_k.ffill().fillna(0)

    @staticmethod
    def calculate_atr(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR).
        """
        # Ensure no NaN values
        high_series = high_series.ffill().fillna(0)
        low_series = low_series.ffill().fillna(0)
        close_series = close_series.ffill().fillna(0)
        
        if close_series.empty or low_series.empty or high_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        high_low = high_series - low_series
        high_close = (high_series - close_series.shift()).abs()
        low_close = (low_series - close_series.shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        
        return atr.ffill().fillna(0)

    @staticmethod
    def calculate_rsi(price_series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a price series.
        """
        # Ensure no NaN values in price series
    
        price_series = price_series.ffill().fillna(0)
        
        
        if price_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        if not isinstance(window, int):
            logging.error(f"RSI calculation error: window parameter must be an integer, got {type(window)}")
            window = int(window)
        
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return rsi.ffill().fillna(0)

    @staticmethod
    def calculate_moving_averages(price_series: pd.Series, short_window: int = 20, long_window: int = 50):
        """
        Calculate short-term and long-term moving averages for a price series.
        """
        # Ensure no NaN values
        price_series = price_series.ffill().fillna(0)
        if price_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        short_ma = price_series.rolling(window=short_window).mean()
        long_ma = price_series.rolling(window=long_window).mean()
        
        return short_ma.ffill().fillna(0), long_ma.ffill().fillna(0)

    @staticmethod
    def calculate_vwap(price_series: pd.Series, volume_series: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        """
        # Ensure no NaN values
        price_series = price_series.ffill().fillna(0)
        volume_series = volume_series.ffill().fillna(0)
        if price_series.empty or volume_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        cumulative_volume = volume_series.cumsum()
        # Avoid division by zero
        cumulative_volume = cumulative_volume.replace(0, np.nan)
        vwap = (price_series * volume_series).cumsum() / cumulative_volume
        vwap.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return vwap.ffill().fillna(0)

    @staticmethod
    def calculate_macd(price_series: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        """
        # Ensure no NaN values
        price_series = price_series.ffill().fillna(0)
        if price_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        short_ema = price_series.ewm(span=short_window, adjust=False).mean()
        long_ema = price_series.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        
        return macd.ffill().fillna(0), signal_line.ffill().fillna(0)

    @staticmethod
    def calculate_bollinger_bands(price_series: pd.Series, window: int = 20, num_std: int = 2):
        """
        Calculate Bollinger Bands for a price series.
        """
        # Ensure no NaN values
        price_series = price_series.ffill().fillna(0)
        if price_series.empty:
            logging.warning("OBV: One or more input series is empty.")
            return pd.Series(dtype='float64')
        
        rolling_mean = price_series.rolling(window).mean()
        rolling_std = price_series.rolling(window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band.ffill().fillna(0), lower_band.ffill().fillna(0)

    @staticmethod
    def calculate_sma(self, price_series: pd.Series, window: int):
        """Calculate the Simple Moving Average (SMA)."""
        return price_series.rolling(window=window).mean()

    def take_rolling_average(self, data: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """
        Calculate the rolling average for a given DataFrame over a specified window size.

        INPUT:
        data   : pd.DataFrame : Data for which to calculate the rolling averages.
        window : int          : The window size over which to calculate the rolling average (default is 7).
        
        OUTPUT:
        pd.DataFrame : A DataFrame with rolling averages.
        """
        # Apply rolling mean to each column in the DataFrame
        rolling_avg_data = data.rolling(window=window, min_periods=1).mean()
        return rolling_avg_data
    
    async def get_news_and_sentiment(self, ticker, bridge = 'USDT', filter="all", news_limit = 10):
        """Fetch news and analyze sentiment for a given ticker using CryptoPanicScraper."""
        try:
            self.news_scraper.topic = ticker.replace(bridge,'')
            self.news_scraper.filter = filter
            self.news_scraper.limit = news_limit
            
            await self.news_scraper.run()  # Run the scraper asynchronously

            # Collect news articles and analyze sentiment
            sentiments = []
            for article in self.news_scraper.data:
                sentiment, confidence = self.news_scraper.analyze_sentiment_with_vader(article["Title"])
                sentiments.append({
                    "title": article["Title"],
                    "sentiment": sentiment,
                    "confidence": confidence
                })

            # Analyze sentiment summary
            bullish_news = len([s for s in sentiments if s['sentiment'] in ['bullish', 'very bullish']])
            bearish_news = len([s for s in sentiments if s['sentiment'] in ['bearish', 'very bearish']])

            # Adjust weights based on sentiment analysis
            sentiment_weight = 0
            if bullish_news > bearish_news:
                sentiment_weight = 0.2  # Increase buy pressure
            elif bearish_news > bullish_news:
                sentiment_weight = -0.2  # Increase sell pressure

            logger.info(f"Sentiment analysis for {ticker}: {bullish_news} bullish, {bearish_news} bearish")
            return sentiment_weight

        except Exception as e:
            logger.error(f"Error fetching news and sentiment for {ticker}: {e}")
            return 0

    async def get_top_gainers(self):
        # Implement logic to return top gainers using the client
        tickers = await self.client.get_ticker()
        sorted_tickers = sorted(tickers, key=lambda x: float(x['priceChangePercent']), reverse=True)
        top_gainers = sorted_tickers[:5]  # Top 5 gainers
        return [{'symbol': t['symbol'], 'change_percent': t['priceChangePercent']} for t in top_gainers]
    
    async def get_top_losers(self):
        tickers = await self.client.get_ticker()
        sorted_tickers = sorted(tickers, key=lambda x: float(x['priceChangePercent']))
        top_losers = sorted_tickers[:5]  # Top 5 losers
        return [{'symbol': t['symbol'], 'change_percent': t['priceChangePercent']} for t in top_losers]

    async def get_market_analysis(self, symbol: str):
        # Example logic for market analysis
        historical_data = await self.get_ticker_price(symbol, days=7, granularity='1h')
        if historical_data.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Calculate some basic indicators as an example
        short_ma, long_ma = self.calculate_moving_averages(historical_data['Close'])
        rsi = self.calculate_rsi(historical_data['Close'])
        
        return {
            "symbol": symbol,
            "short_ma": short_ma.iloc[-1] if not short_ma.empty else None,
            "long_ma": long_ma.iloc[-1] if not long_ma.empty else None,
            "rsi": rsi.iloc[-1] if not rsi.empty else None,
        }

    async def get_market_performance(self, symbol: str):
        historical_data = await self.get_ticker_price(symbol, days=1, granularity="1m")
        if historical_data.empty:
            return {"error": f"No performance data available for {symbol}"}
        
        return {
            "symbol": symbol,
            "price": historical_data['Close'].iloc[-1],
            "volume": historical_data['Volume'].iloc[-1]
        }
    
    async def get_historical_data(self, symbol: str, days: int = 7, granularity: str = "1h"):
        historical_data = await self.get_ticker_price(symbol, days, granularity)
        if historical_data.empty:
            return {"error": f"No historical data available for {symbol}"}
        
        return historical_data.to_dict(orient='records')
    
    async def get_recommendation(self, symbol: str):
        # Fetch historical data and use your prediction logic
        historical_data = await self.get_ticker_price(symbol, days=30, granularity="1h")
        if historical_data.empty:
            return {"error": f"No data available to make a recommendation for {symbol}"}
        
        # Example: use mock indicator data for prediction
        rsi = self.calculate_rsi(historical_data['Close'])
        short_ma, long_ma = self.calculate_moving_averages(historical_data['Close'])
        
        # Mock order book data for prediction
        bids, asks = await self.get_order_book(symbol)
        
        # Weights for indicators (customizable)
        weights = {
            'rsi': 0.2,
            'ma': 0.3,
            'vwap': 0.2,
            'macd': 0.1,
            'bbands': 0.05,
            'atr': 0.1,
            'stochastic': 0.05,
            'obv': 0.1,
        }
        
        # Predict price and action (buy/sell/hold)
        recommended_price, action = self.predict_price(
            rsi=rsi,
            bids=bids,
            asks=asks,
            short_ma=short_ma,
            long_ma=long_ma,
            vwap=None,  # Add VWAP logic here
            macd=None,  # Add MACD logic here
            macd_signal=None,
            bb_upper=None,  # Add Bollinger Bands logic here
            bb_lower=None,
            atr=None,
            stochastic=None,
            obv=None,
            weights=weights
        )
        
        return {
            "symbol": symbol,
            "action": action,
            "recommended_price": recommended_price
        }

    async def run(self):
        """
        Main logic for running the bot.
        Fetches data, performs analysis, and executes predictions.
        """
        await self.initialize_client()

        bridge = 'USDT'
        exchange_info = await self.client.get_exchange_info()
        full_coin_list = [
            s['symbol'][:-len(bridge)] for s in exchange_info['symbols']
            if s['symbol'].endswith(bridge)
        ]
        forbidden_words = [
            'DOWN', 'UP', 'BULL', 'BEAR', 'DWN', 'UP'
        ]
        full_coin_list = [
            coin for coin in full_coin_list
            if not any(word in coin for word in forbidden_words)
        ]
        full_coin_list = list(set(full_coin_list))  # Remove duplicates
        full_coin_list.sort()

        # For testing purposes, limit to coins with high volume
        cointickers = [coin + bridge for coin in full_coin_list]
        cointickers = cointickers[:5]

        # Fetch data for correlation calculations
        day_data = await self.get_price_data(cointickers, 1, "1m")
        week_data = await self.get_price_data(cointickers, 7, "1h")
        two_week_data = await self.get_price_data(cointickers, 14, "2h")

        # Perform percentage change for correlation
        numeric_columns_day = day_data.columns.difference(['date'])
        numeric_columns_week = week_data.columns.difference(['date'])
        numeric_columns_two_week = two_week_data.columns.difference(['date'])
        
        day_data[numeric_columns_day] = day_data[numeric_columns_day].apply(pd.to_numeric, errors='coerce')
        week_data[numeric_columns_week] = week_data[numeric_columns_week].apply(pd.to_numeric, errors='coerce')
        two_week_data[numeric_columns_two_week] = two_week_data[numeric_columns_two_week].apply(pd.to_numeric, errors='coerce')

        day_data.dropna(inplace=True)  
        week_data.dropna(inplace=True)
        two_week_data.dropna(inplace=True)

        # Calculate percentage change after ensuring numeric values
        day_data_pct = day_data[numeric_columns_day].pct_change()
        week_data_pct = week_data[numeric_columns_week].pct_change()
        two_week_data_pct = two_week_data[numeric_columns_two_week].pct_change()        
       
        # Calculate rolling averages
        RA_day_data = self.take_rolling_average(day_data_pct)
        RA_week_data = self.take_rolling_average(week_data_pct)
        RA_2week_data = self.take_rolling_average(two_week_data_pct)

        # Calculate correlations
        day_corr = RA_day_data.corr()
        week_corr = RA_week_data.corr()
        two_week_corr = RA_2week_data.corr()
        
        list_size = len(cointickers)
        
        # Pick coins based on correlations
        coinlist = self.pick_coins(cointickers, day_corr, week_corr, two_week_corr, list_size)
        logger.info(f"Selected coins for analysis: {coinlist}")

        # Initialize dictionaries to store buy and sell recommendations
        recommended_buy_dic = {}
        recommended_sell_dic = {}

        # Proceed with your analysis on the selected coins
        for ticker in coinlist:
            try:
                # Technical Analysis
                # Check if required columns are present before proceeding
                required_columns = [
                    f'Price_{ticker}', f'High_{ticker}', f'Low_{ticker}',
                    f'Close_{ticker}', f'Volume_{ticker}'
                ]

                if not all(col in day_data.columns for col in required_columns):
                    logger.warning(f"Data for {ticker} is incomplete. Skipping analysis.")
                    continue

                # Prepare data
                price_data = day_data[f'Price_{ticker}']
                volume_data = day_data[f'Volume_{ticker}']
                high_data = day_data[f'High_{ticker}']
                low_data = day_data[f'Low_{ticker}']
                close_data = day_data[f'Close_{ticker}']

                # Check if there is sufficient data for indicator calculation
                if not self.has_sufficient_data(price_data, volume_data):
                    logger.warning(
                        f"Insufficient data for {ticker}. Skipping."
                    )
                    continue

                # Drop any remaining NaNs
                price_data = price_data.dropna()
                volume_data = volume_data.dropna()
                high_data = high_data.dropna()
                low_data = low_data.dropna()
                close_data = close_data.dropna()

                # Calculate indicators
                rsi = self.calculate_rsi(price_data)
                short_ma, long_ma = self.calculate_moving_averages(price_data)
                vwap = self.calculate_vwap(price_data, volume_data)
                macd, macd_signal = self.calculate_macd(price_data)
                bb_upper, bb_lower = self.calculate_bollinger_bands(
                    price_data
                )
                atr = self.calculate_atr(
                    high_data, low_data, close_data
                )
                stochastic = self.calculate_stochastic_oscillator(
                    high_data, low_data, close_data
                )
                obv = self.calculate_obv(close_data, volume_data)

                # Ensure indicators have enough data
                indicators = {
                    'rsi': rsi,
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'vwap': vwap,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'atr': atr,
                    'stochastic': stochastic,
                    'obv': obv
                }

                for name, series in indicators.items():
                    if series.empty or len(series) < 1:
                        logger.warning(
                            f"{name} Series is empty or has insufficient "
                            f"data for {ticker}. Skipping."
                        )
                        raise ValueError(f"{name} insufficient data")

                # Fetch order book
                bids, asks = await self.get_order_book(ticker)
                if bids.empty or asks.empty:
                    logger.warning(
                        f"Order book for {ticker} is empty or failed to fetch."
                    )
                    continue

                
                # Define weights for decision-making
                weights = {
                    'rsi': 0.2,
                    'ma': 0.1,
                    'vwap': 0.1,
                    'macd': 0.1,
                    'bbands': 0.05,
                    'atr': 0.1,
                    'stochastic': 0.1,
                    'obv': 0.1,
                }

                # Perform prediction
                recommended_price, action = self.predict_price(
                    rsi=rsi,
                    bids=bids,
                    asks=asks,
                    short_ma=short_ma,
                    long_ma=long_ma,
                    vwap=vwap,
                    macd=macd,
                    macd_signal=macd_signal,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    atr=atr,
                    stochastic=stochastic,
                    obv=obv,
                    weights=weights
                )

                # Ensure recommended_price is valid before logging
                if recommended_price is not None:
                    try:
                        price_float = float(recommended_price)
                        if action == "Sell":
                            recommended_sell_dic[ticker] = price_float
                        else:
                            recommended_buy_dic[ticker] = price_float

                        logger.info(
                            f"{action} signal for {ticker} at price {price_float}"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid recommended price: {recommended_price} ({e})")
                        logger.info(f"No clear signal for {ticker}. Holding position.")
                else:
                    logger.info(f"No clear signal for {ticker}. Holding position.")

                # Generate a detailed report
                #self.generate_report(ticker, indicators)

                # Optional: Perform backtesting for this ticker
                # historical_data = await self.get_ticker_price(
                #     ticker, days=30, granularity="1h"
                # )
                # if not historical_data.empty:
                #     historical_data.rename(columns={
                #         'High': 'High',
                #         'Low': 'Low',
                #         'Close': 'Close',
                #         'Volume': 'Volume'
                #     }, inplace=True)
                #     historical_data['symbol'] = ticker
                #     accuracy, avg_profit = self.backtest_strategy(
                #         historical_data, weights
                #     )
                #     logging.info(
                #         f"Backtesting for {ticker} - Accuracy: "
                #         f"{accuracy:.2f}%, Avg Profit: {avg_profit:.4f}"
                #     )

            except Exception as e:
                logger.error(
                    f"Error while analyzing {ticker}: {e}"
                )
                logger.error(traceback.format_exc())

        logging.info(f'Recommended buy list: {recommended_buy_dic}')
        logging.info(f'Recommended sell list: {recommended_sell_dic}')

        # Fetch and log 24-hour trade volume for selected coins
        await self.get_trade_volume(coinlist)

        # Close client after all tasks
        await self.close_client()
        logger.info("Client session closed.")

# Run the bot
if __name__ == "__main__":
    api_key = ""
    api_secret = ""
    
    bot = TradingBot(api_key, api_secret)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logging.info("Bot interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error("Stack Trace:", exc_info=True)
    
