from enum import Enum


class DataCol(Enum):
    OpenTime = 0
    Open = 1
    High = 2
    Low = 3
    Close = 4
    Type = 5


ColName = {
    DataCol.OpenTime: 'open_time',
    DataCol.Open: 'open',
    DataCol.High: 'high',
    DataCol.Low: 'low',
    DataCol.Close: 'close',
    DataCol.Type: 'type'
}
