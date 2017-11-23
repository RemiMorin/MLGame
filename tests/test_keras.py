import unittest
import keras_example
import math


class TestNeuron(unittest.TestCase):

    def setUp(self):

        pass

    def test_init(self):
        x_test, y_test = keras_example.getTrain(10)

        for i in range(0,len(y_test)):
            print( str(x_test[i]) + " " + str(y_test[i]))

    def test_trained_network(self):
        model = keras_example.get_trained_model(1000000)
        x_test, y_test = keras_example.getTrain(10)
        predictions = model.predict(x_test)
        for i in range(0,len(y_test)):
            print("%.5f\t%.5f" % (predictions[i],y_test[i]))

    def test_tanh(self):
        x_test, y_test = keras_example.getTrain(10)
        x_test.append([-0.01,-1])
        x_test.append([ 0.01,-1])
        x_test.append([-0,-1])
        x_test.append([-0.01,1])
        y_test.append(1)
        y_test.append(1)
        y_test.append(1)
        y_test.append(1)
        for i in range(0,len(y_test)):
            print("%.5f\t%.5f" % (math.atan2(x_test[i][0],x_test[i][1])/ math.pi,y_test[i]))


if __name__ == '__main__':
    unittest.main()