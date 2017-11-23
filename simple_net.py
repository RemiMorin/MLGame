import sys, getopt
from keras_example import getTrain
import random
import math
from shark_game import Character, getCharacters

def sigmoid(value):
    if (value < -100):
        value = -100
    if (value > 100):
        value = 100
    #if (value < 1.e-20 and value > -1.e-20):

    try:
        return 1. / (1. + math.exp(-value))
    except:
        print ("shiiiiiiiiiiiiiiiiiiiiiiiiite")
        print (value)
        return 0

def transfer_derivative(output):
	return output * (1.0 - output)

def tanh_transfer_derivative(output):
    return 1 - math.pow(math.tanh(output),2.)

def computeError(prediction,expected):
    error =  expected - prediction * tanh_transfer_derivative(prediction)
    return error




def randInit():
    return random.uniform(-1.,1.)

def initArrayRandom(length):
    return [randInit() for i in range(0,length)]

class NeuralNetworkElement(object):

    def getResult(self):
        pass

    def compute(self):
        pass

class PlaceHolderElement(NeuralNetworkElement):

    def setValue(self, value):
        self.value = value

    def getResult(self):
        return self.value

class Neuron(NeuralNetworkElement):
    def __init__(self,inputs):
        self.inputs = inputs
        self.weights = initArrayRandom(len(inputs))
        self.momentum = [.0] * len(inputs)
        self.bias = randInit()
        self.bias_momentum = 0.0

    def compute(self):
        self.result = 0
        for i in range(0,len(self.inputs)):
            self.result += self.inputs[i].getResult() * self.weights[i]
        self.result = math.tanh(self.result + self.bias)
        return self.result

    def getResult(self):
        return self.result


class Layer(object):
    def __init__(self,inputs,layerSize):
        self.neurons = [Neuron(inputs) for i in range (0,layerSize)]

    def compute(self):
        for neuron in self.neurons:
            neuron.compute()

    def getResults(self):
        return [neuron.getResult() for neuron in self.neurons]

    def printLayer(self):
        print ("********************* layer **************")
        for neuron in self.neurons:
            print(("neuron %.2f [" % neuron.bias) + "".join(["\t%.2f " % weight for weight in neuron.weights ]) + "]" )


class Kernel(object):
    def __init__(self,inputs,dimensions):
        previous_layer = inputs
        self.layers = []
        for layer_dimension in dimensions:
            self.layers.append(Layer(previous_layer,layer_dimension))
            previous_layer = self.layers[-1].neurons

    def compute(self):
        for layer in self.layers:
            layer.compute()

    def backPropagate(self,error,learning_rate):
        layer = self.layers[-1]
        for i in range(len(layer.neurons)):
            layer.neurons[i].error = error[i] * tanh_transfer_derivative(layer.neurons[i].result)
        previous_layer = layer
        for layer in reversed(self.layers[:-1]):
            for neuron_index in range(len(layer.neurons)):
                neuron = layer.neurons[neuron_index]
                neuron.error = 0.0
                for neuron_previous_layer in previous_layer.neurons:
                    neuron.error = neuron.error + (neuron_previous_layer.error * neuron_previous_layer.weights[neuron_index])
                neuron.error = neuron.error  * tanh_transfer_derivative(neuron.result)
            previous_layer = layer
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.momentum[i] = (neuron.momentum[i] * 0.5) +  (learning_rate * neuron.error * neuron.inputs[i].getResult())
                    neuron.weights[i] = neuron.weights[i] + neuron.momentum[i]
                neuron.bias_momentum = (neuron.bias_momentum * 0.0) + (learning_rate * neuron.error)
                neuron.bias = neuron.bias + neuron.bias_momentum

    def printKernel(self):
        for layer in self.layers:
            layer.printLayer()

    def getResult(self):
        return self.layers[-1].getResults()


def main(argv):
    diffx = PlaceHolderElement()
    diffy = PlaceHolderElement()


    kernel = Kernel([diffy,diffx],[10,10,10,1])
    kernel.layers[-1].neurons[0].bias = 0

    error_sum = 0.0
    learning_rate = 0.00001
    x_train, y_train = getTrain(500000)

    for j in range (0,1000):
        print ("epoch %d" % j)
        print("learning rate " + str(learning_rate))

        for i in range(0,len(y_train)):

            diffx.setValue(x_train[i][0])
            diffy.setValue(x_train[i][1])

            kernel.compute()

            error = [computeError(result,y_train[i]) for result in kernel.getResult() ]
            error_sum = error_sum + abs(error[0])
            if(i%10000 == 9999):
                print(error_sum/10000)
                error_sum = 0.0
            kernel.backPropagate(error,learning_rate)

        kernel.printKernel()
        learning_rate = learning_rate * 0.9



if __name__ == "__main__":
    main(sys.argv[1:])
