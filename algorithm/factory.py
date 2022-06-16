from enum import Enum
from algorithm.lstm import LSTM
from algorithm.rnn import RNN
from algorithm.xgboost import XGBoost


class Algorithm(Enum):
    LTSM = 0
    RNN = 1
    XGBOOST = 2


class AlgorithmFactory:
    @staticmethod
    def get(algorithm_enum: Algorithm):
        if algorithm_enum == Algorithm.LTSM:
            return LSTM()
        elif algorithm_enum == Algorithm.RNN:
            return RNN()
        elif algorithm_enum == Algorithm.XGBOOST:
            return XGBoost()
        return None

