from algorithm.base_algorithm import BaseAlgorithm
from algorithm.base_darts import BaseDartsAlgorithm
from darts.models.forecasting.transformer_model import TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries


class Transformer(BaseDartsAlgorithm):
    def __init__(self):
        model = TransformerModel(
            dropout=0,
            batch_size=8,
            n_epochs=1,
            model_name="Transformer",
            # log_tensorboard=True,
            random_state=42,
            input_chunk_length=60,
            output_chunk_length=1,
            force_reset=True,
            save_checkpoints=True,
        )
        super().__init__(model)

