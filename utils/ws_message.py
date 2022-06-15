import json
import pandas as pd


def format_binance_message(message):
    if message is None:
        return None, False, False

    # print(message)
    data = message['data']
    data = json.loads(data)

    candle = data['k']
    row = pd.DataFrame({
        'open_time': pd.to_datetime(candle['t'], unit='ms'),
        'open': float(candle['o']),
        'low': float(candle['l']),
        'high': float(candle['h']),
        'close': float(candle['c'])
    }, index=[0])
    is_closed = candle['x']

    return row, is_closed, True

