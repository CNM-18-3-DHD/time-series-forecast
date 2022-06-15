from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# From working directory
MODEL_LOCATION = 'mlmodels/LSTM_ETHUSDT_1m_close.h5'
model = load_model(MODEL_LOCATION)
def get_scaler():
    return MinMaxScaler(feature_range=(0, 1))



class LSTMCloseAlgorithm:
    def __init__(self):
        self.scaler = get_scaler()

    def fit(self, data):
        new_data = pd.DataFrame(index=range(0, len(data) + 1), columns=['open_time', 'close'])
        # print(len(new_data))
        for i in range(0, len(data)):
            new_data["open_time"][i] = data['open_time'][i]
            new_data["close"][i] = data['close'][i]

        # Add next timeframe: x[n] = x[n-1] + (x[n-1] - x[n-2])
        next_timeframe = data["open_time"][len(data) - 1] + (
                data["open_time"][len(data) - 1] - data["open_time"][len(data) - 2]
        )
        # print(data["open_time"][len(data) - 1])
        # print(next_timeframe)
        new_data["open_time"][len(data)] = next_timeframe
        new_data["close"][len(data)] = new_data["close"][len(data) - 1]

        x_value = []
        for i in range(941, len(new_data)):
            x_value.append(new_data["open_time"][i])

        new_data.index = new_data.open_time
        new_data.drop("open_time", axis=1, inplace=True)
        self.scaler = get_scaler()
        dataset = new_data.values

        # train = dataset[0:941, :]
        valid = dataset[941:, :]

        scaled_data = self.scaler.fit_transform(dataset)

        # Unused, not needed
        # x_train, y_train = [], []
        #
        # for i in range(60, len(train)):
        #     x_train.append(scaled_data[i - 60:i, 0])
        #     y_train.append(scaled_data[i, 0])

        # x_train = np.array(x_train)
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)

        x_test = []
        for i in range(60, inputs.shape[0]):
            x_test.append(inputs[i - 60:i, 0])
        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return [x_test, x_value]

    def predict(self, current_data):
        [predict_x, x_value] = self.fit(current_data)
        predict_y = model.predict(predict_x)
        predict_y = self.scaler.inverse_transform(predict_y)
        # Covert [[num1], [num1],...] => [num1, num2,...]
        predict_y = np.asarray(predict_y).reshape(-1)
        return [x_value, predict_y]


