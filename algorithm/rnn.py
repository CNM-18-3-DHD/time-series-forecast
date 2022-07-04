from algorithm.base_darts import BaseDartsAlgorithm
from darts.models.forecasting.rnn_model import RNNModel


class RNN(BaseDartsAlgorithm):
    def __init__(self):
        model = RNNModel(
            model="RNN",
            # Size for feature maps for each hidden RNN layer
            hidden_dim=20,
            # Fraction of neurons affected by Dropout
            dropout=0,
            # Number of time series (input and output sequences) used in each training pass
            batch_size=4,
            # Number of epochs over which to train the model
            n_epochs=1,
            model_name="RNN",
            # log_tensorboard=True,
            # Control the randomness of the weights initialization
            random_state=42,
            # Number of past time steps that are fed to the forecasting module at prediction time
            input_chunk_length=60,
            # If set to True, any previously-existing model with the same name will be reset
            force_reset=True,
            # Whether or not to automatically save the untrained model and checkpoints from training
            save_checkpoints=True,
        )
        super().__init__(model)
