import unittest
import simple_net
import math

class TestNeuron(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        elements = [simple_net.PlaceHolderElement() for i in range(0, 10)]
        for element in elements:
            element.setValue(1.)
        n = simple_net.Neuron(elements);
        n.compute()
        print(n.getResult())

    def test_kernel(self):
        elements = [simple_net.PlaceHolderElement() for i in range(0, 5)]
        kernel = simple_net.Kernel(elements,[5,4,3,5])
        for element in elements:
            element.setValue(12.)
        kernel.compute()
        print(kernel.getResult())

    def test_kernel_creation(self):
        elements = [simple_net.PlaceHolderElement() for i in range(0, 5)]
        kernel = simple_net.Kernel(elements,[5,4,3,5])
        self.assertEqual(len(kernel.layers), 4)
        self.assertEqual(len(kernel.layers[0].neurons), 5)
        self.assertEqual(len(kernel.layers[1].neurons), 4)
        self.assertEqual(len(kernel.layers[2].neurons), 3)
        self.assertEqual(len(kernel.layers[3].neurons), 5)
        for neuron in kernel.layers[0].neurons:
            self.assertEqual(len(neuron.weights), 5)
        for neuron in kernel.layers[1].neurons:
            self.assertEqual(len(neuron.weights), 5)
        for neuron in kernel.layers[2].neurons:
            self.assertEqual(len(neuron.weights), 4)
        for neuron in kernel.layers[3].neurons:
            self.assertEqual(len(neuron.weights), 3)

    def test_sigmoid(self):
        for i in range(-11,11):
            simple_net.sigmoid(i)
        simple_net.sigmoid(1.e-200)
        simple_net.sigmoid(1.e-300)
        simple_net.sigmoid(-1.e-200)
        simple_net.sigmoid(0)



    def computeError(self,expected_error,prediction,expected):
        print("prediction " + str(prediction) + "\texpected "+str(expected)+"\tExpected error " + str(expected_error) + "\t||| result " + str(simple_net.computeError(prediction,expected)))

    def test_compute_error(self):
        a = []

        print(a)


if __name__ == '__main__':
    unittest.main()