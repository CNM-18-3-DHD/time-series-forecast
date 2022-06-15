import csv
from binance.client import Client

DEFAULT_TRADE_SYMBOL = "ETHUSDT"

symbol = DEFAULT_TRADE_SYMBOL
interval = Client.KLINE_INTERVAL_1MINUTE
client = Client()   # No API key/secret needed for this type of call

columns = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'ignore'
]

klines = client.get_historical_klines(symbol, interval)

with open(f'data/{symbol}-{interval}-DATA.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(columns)
    write.writerows(klines)
