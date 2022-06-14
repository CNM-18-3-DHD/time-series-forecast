from algorithm.base_algorithm import BaseAlgorithm
from darts.models.forecasting.rnn_model import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries


class RNN(BaseAlgorithm):
    def __init__(self):
        self.model = RNNModel(
            model="RNN",
            hidden_dim=20,
            dropout=0,
            batch_size=4,
            n_epochs=1,
            # optimizer_kwargs={"lr": 1e-3},
            model_name="LSTM",
            # log_tensorboard=True,
            random_state=42,
            input_chunk_length=60,
            force_reset=True,
            save_checkpoints=True,
        )
        self.scaler = None

    def fit(self, df):
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
