import tensorflow as tf

from tensorflow import keras


class BTCLSTMModel(keras.Model):

    def __init__(self, lstm_units_1: int = 32, lstm_units_2 = 64, name: str = 'BTC_LSTM_Model', **kwargs):
        super().__init__(name = name, **kwargs)

        self.lstm_1 = keras.layers.LSTM(units = lstm_units_1, return_sequences=True)
        self.lstm_2 = keras.layers.LSTM(units = lstm_units_2, return_sequences=False)
        self.dense = keras.layers.Dense(1)

    def call(self, inputs):

        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        return self.dense(x)