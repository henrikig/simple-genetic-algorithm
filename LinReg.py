import random
import time
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class LinReg:
    def __init__(self):
        pass

    def train(self, data, y):
        model = LinearRegression().fit(data, y)
        return model

    def get_fitness(self, x, y, random_state=42):
        if random_state == 0:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
        model = self.train(x_train, y_train)
        predictions = model.predict(x_test)
        error = sqrt(mean_squared_error(predictions, y_test))

        return error

    def get_columns(self, x, bitstring):
        # Function to filter data based on a bitstring
        indexes = []
        for i, s in enumerate(bitstring):
            if s == "0":
                indexes.append(i)
        arr = np.asarray(x)
        arr = np.delete(arr, indexes, axis=1)
        return arr


if __name__ == "__main__":
    lr = LinReg()
    df = pd.read_csv("house_data.csv")

    data = df[df.columns[3:]]
    values = df[df.columns[2]]

    bitstring = str(bin(random.getrandbits(18)))[2:]
    t = time.perf_counter()
    fitness2 = lr.get_fitness(data, values)
    t2 = time.perf_counter()
    print("time", t2 - t)

    data = lr.get_columns(data, bitstring)
    fitness = lr.get_fitness(data, values)

    print(fitness)
    print(fitness2)
