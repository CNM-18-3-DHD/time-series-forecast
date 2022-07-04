from algorithm.base_darts import BaseDartsAlgorithm
from darts.models.forecasting.transformer_model import TransformerModel


class Transformer(BaseDartsAlgorithm):
    def __init__(self):
        model = TransformerModel(
            # Fraction of neurons affected by Dropout
            dropout=0,
            # Number of time series (input and output sequences) used in each training pass
            batch_size=8,
            # Number of epochs over which to train the model
            n_epochs=1,
            model_name="Transformer",
            # log_tensorboard=True,
            # Control the randomness of the weights initialization
            random_state=42,
            # Number of time steps to be input to the forecasting module
            input_chunk_length=60,
            # Number of time steps to be output by the forecasting module
            output_chunk_length=1,
            # If set to True, any previously-existing model with the same name will be reset
            force_reset=True,
            # Whether or not to automatically save the untrained model and checkpoints from training
            save_checkpoints=True,
        )
        super().__init__(model)

