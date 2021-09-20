#!/usr/bin/env python3
""" Bitcoin price time series prediction project"""
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import preprocess_data


X_train, y_train = preprocess_data.X_train, preprocess_data.y_train
X_test, y_test = preprocess_data.X_test, preprocess_data.y_test

model = k.models.Sequential()

model.add(k.layers.LSTM(units=50, return_sequences=True,
                        input_shape=(X_train.shape[1], 1)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.LSTM(units=50, return_sequences=True))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.LSTM(units=50, return_sequences=True))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.LSTM(units=50))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
train_dataset = train_dataset.shuffle(100).batch(32)
test_dataset = test_dataset.batch(32)
model.fit(train_dataset, epochs=5, steps_per_epoch=5)

pred = model.predict(test_dataset, steps=5)
model.evaluate(test_dataset, steps=5)

sc = preprocess_data.sc
p = sc.inverse_transform(pred)
y = sc.inverse_transform(y_test)

plt.figure(figsize=(20, 12))
plt.plot(y, color='orange', label='True')
plt.plot(p, color='green', label='Predicted')
plt.title('Price prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()