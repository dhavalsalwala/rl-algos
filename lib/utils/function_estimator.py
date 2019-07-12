from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


class ANN:
    def __init__(self, nS, nA, learning_rate=0.001):

        self.nS = nS
        self.nA = nA
        self.learning_rate = learning_rate
        self.model = self.build()

    def build(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.nS, activation='relu'))
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(self.learning_rate))
        return model

    def train(self, x, y, batch_size=64, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size, epochs=epochs, verbose=verbose)

    def predict(self, state):
        if state.ndim == 1:
            return self.predict(state.reshape(1, self.nS)).flatten()
        else:
            return self.model.predict(state)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
