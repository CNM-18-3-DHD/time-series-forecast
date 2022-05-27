from enum import Enum
from binance import Client


class TimeframeTypes(Enum):
    TF_1MIN = 1
    TF_3MIN = 2
    TF_5MIN = 3


Timeframes = {
    TimeframeTypes.TF_1MIN: Client.KLINE_INTERVAL_1MINUTE,
    TimeframeTypes.TF_3MIN: Client.KLINE_INTERVAL_3MINUTE,
    TimeframeTypes.TF_5MIN: Client.KLINE_INTERVAL_5MINUTE,
}

TimeframesGap = {
    TimeframeTypes.TF_1MIN: 60000,     # 60s * 1000 (ms)
    TimeframeTypes.TF_3MIN: 180000,    # 3 * 60s * 1000 (ms)
    TimeframeTypes.TF_5MIN: 300000,    # 5 * 60s * 1000 (ms)
}
