from algorithm.base_algorithm import BaseAlgorithm
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
import pandas as pd


# Darts models base adapter
class BaseDartsAlgorithm(BaseAlgorithm):
    def __init__(self, model):
        self.model = model
        self.scaler = None
        self.timestamp_gap = None

    def fit(self, df):
        self.timestamp_gap = df['open_time'].iloc[-1] - df['open_time'].iloc[-2]
        df_formatted = df[['open_time', 'close']].copy()
        series = TimeSeries.from_dataframe(df_formatted, time_col='open_time')
        self.scaler = Scaler()
        series_transformed = self.scaler.fit_transform(series)
        self.model.fit(
            series_transformed
        )

    def predict(self, current_data=None, n=1):
        series_predict = self.model.predict(n)
        series_predict = self.scaler.inverse_transform(series_predict)
        df_predict = series_predict.pd_dataframe()
        df_predict = df_predict.reset_index()  # Move open_time to a column
        return df_predict

    def predict_step(self, step=1, context=None):
        df_predict = self.predict(n=step)
        # df_predict = df_predict.tail(1)  # get last step of prediction, old
        # Return next timestamp step prediction, cheekily
        last_row_df_predict = df_predict.iloc[-1]  # returns dict
        last_row_ctx = context.iloc[-1]  # returns dict
        predict_timestamp = last_row_ctx['open_time'] + self.timestamp_gap
        df_predict = pd.DataFrame({
            'open_time': predict_timestamp,
            'close': last_row_df_predict['close']
        }, index=[0])
        # Return next timestamp step prediction, cheekily
        return df_predict
