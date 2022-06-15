from keras.models import Sequential
from keras.layers import LSTM, Dense


def build_ltsm(x_train_data, y_train_data):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    return lstm_model
