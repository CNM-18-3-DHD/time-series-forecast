from algorithm.base_algorithm import BaseAlgorithm
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    cols = list()
    df_x = df[['close']].copy()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df_x.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df_x.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


class XGBoost(BaseAlgorithm):
    def __init__(self):
        self.model = XGBRegressor()
        self.timestamp_gap = None

    def fit(self, df):
        self.timestamp_gap = df['open_time'].iloc[-1] - df['open_time'].iloc[-2]
        series = series_to_supervised(df=df)
        train = np.asarray(series)
        # split into input and output columns
        train_x, train_y = train[:, :-1], train[:, -1]
        self.model.fit(train_x, train_y)

    def predict(self, current_data=None, n=1):
        last_row = current_data.iloc[-1]  # returns dict
        predict_x = last_row['close']
        predict_timestamp = last_row['open_time'] + self.timestamp_gap
        predict_y = self.model.predict(np.asarray([predict_x]))
        df_predict = pd.DataFrame({
            'open_time': predict_timestamp,
            'close': predict_y
        }, index=[0])
        # print(df_predict)
        return df_predict

    def predict_step(self, step=1, context=None):
        return self.predict(current_data=context, n=step)
