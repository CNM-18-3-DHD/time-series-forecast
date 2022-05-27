import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime
import src.utils.mlmodels as mlmodels
from src.utils.configs import get_scaler

symbol = "ETHUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE
feature = "close"
model = "LSTM"
save_filename = f"../mlmodels/{model}_{symbol}_{interval}_{feature}.h5"
print(save_filename)

scaler = get_scaler()

# Load data
df = pd.read_csv(f'data/{symbol}-{interval}-DATA.csv')
df["open_time"] = pd.to_datetime(df.open_time, unit="ms")
print(df)
df.index = df["open_time"]

data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=["open_time", "close"])

for i in range(0, len(df)):
    new_dataset["open_time"][i] = data["open_time"][i]
    new_dataset["close"][i] = data["close"][i]

# Normalize filtered dataset
new_dataset.index = new_dataset.open_time
new_dataset.drop("open_time", axis=1, inplace=True)

final_dataset = new_dataset.values
train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

model = mlmodels.build_ltsm(x_train_data, y_train_data)                 # Build & train model
model.save(save_filename)          # Save


# Get test data
inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

test_data_x = []
for i in range(60, inputs_data.shape[0]):
    test_data_x.append(inputs_data[i - 60:i, 0])
test_data_x = np.array(test_data_x)
test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], test_data_x.shape[1], 1))
# Get test data
prediction_closing_price = model.predict(test_data_x)   # a.k.a test_data_y_predict
prediction_closing_price = scaler.inverse_transform(prediction_closing_price)

train_data = new_dataset[:987]
valid_data = new_dataset[987:]
valid_data['predictions'] = prediction_closing_price
plt.plot(train_data["close"])
plt.plot(valid_data[['close', "predictions"]])
plt.show()
