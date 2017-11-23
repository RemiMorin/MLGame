import sys
import random
import keras
import math
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

def randomArray(length,cap):
    return [random.uniform(-cap,cap)/cap for i in range(0,length)]

def random2DArray(x,count,cap):
    return [randomArray(x,cap) for i in range(0,count)]

def getTrain(count):
    x = random2DArray(2,count,100)
    y = [ math.atan2(val[0],val[1])/ math.pi for val in x ]
    return x,y

def main(argv):

    model = get_trained_model(10000)
    test_x, test_y = getTrain(128)
    score = model.evaluate(test_x, test_y, batch_size=128)
    x = np.array([[1, 2]])
    print (model.predict(x))

    print(score)

def get_trained_model(train_set_size):
    train_x,train_y = getTrain(train_set_size)

    model = Sequential()
    model.add(Dense(units=10, activation='softmax', input_dim=2))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss=keras.losses.MSE,
                  optimizer='rmsprop')

    model.fit(train_x,train_y, epochs=10, batch_size=1000)
    return model




if __name__ == "__main__":
    main(sys.argv[1:])
