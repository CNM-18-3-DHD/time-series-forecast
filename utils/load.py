from binance import Client
import pandas as pd

DEFAULT_SYMBOL = "ETHUSDT"

COL_OPEN_TIME = "open_time"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_TYPE = "type"

client = Client()

# Functions =================================


def get_empty_data():
    return {
        COL_OPEN_TIME: [],
        COL_OPEN: [],
        COL_HIGH: [],
        COL_LOW: [],
        COL_CLOSE: [],
    }


def load_symbol_data():
    tickers = client.get_all_tickers()
    data_symbols = []
    # Format
    for ticker in tickers:
        data_symbols.append(ticker["symbol"])
    return data_symbols


ws_base_url = 'wss://stream.binance.com:9443'


def get_ws_url(symbol=DEFAULT_SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE):
    lower_symbol = symbol.lower()
    return f"{ws_base_url}/ws/{lower_symbol}@kline_{interval}"


def load_data(symbol=DEFAULT_SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000, start_str=None):
    data = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_str=start_str
    )

    formatted = get_empty_data()
    for row in data:
        formatted[COL_OPEN_TIME].append(pd.to_datetime(row[0], unit='ms'))
        formatted[COL_OPEN].append(float(row[1]))
        formatted[COL_HIGH].append(float(row[2]))
        formatted[COL_LOW].append(float(row[3]))
        formatted[COL_CLOSE].append(float(row[4]))

    return formatted


# Functions =================================
