from algorithm.base_darts import BaseDartsAlgorithm
from darts.models.forecasting.rnn_model import RNNModel


class LSTM(BaseDartsAlgorithm):
    def __init__(self):
        model = RNNModel(
            model="LSTM",
            hidden_dim=20,
            dropout=0,
            batch_size=4,
            n_epochs=1,
            model_name="LSTM",
            # log_tensorboard=True,
            random_state=42,
            input_chunk_length=60,
            force_reset=True,
            save_checkpoints=True,
        )
        super().__init__(model)
