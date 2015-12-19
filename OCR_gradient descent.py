import numpy as np


# initialize object
class Network (object):

    def __init__(self, sizes):
        # index of sizes represent layer number, each value represents num of neurons within each layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # np.random.randn gives a standard normal distribution, enable the stochastic gradient descent
        # each recipient neuron will need a bias term
        # biases will have layer number -1 array, each array values for each neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # Weights will be matrix determining by the number of input and output neurons of the network
        # Zip combine the input elements with the same index as tuple
        # e.g if a = [1,2,3], b = [3,4,5], then zip(a[:-1],b[1:]) will give [(1,4),(2,5)]

    def feedforward(self, a):
        # Return the output of the network if "a" is provided
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        # training data is a list of tuples "(x,y)" represnting the training input and the desired outputs.
        # If "test_data" is provided, the network will be evaluated against the test data after each epoch
        # and priting partial result
        # eta is the learning rate
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)  # generate the random shuffled training_data
            mini_batchs = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                # if test_data is provided, the network will be evaluated after each epoch of training
                print ("Epoch{0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch{0} complete".format(j))

    # Update the network's weights and biases by applying gradient descent using backpropagation to a single mini_batch.
    # The "mini_batch" is a list of tuples "(x,y)", and the "eta" is the learning rate

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # for array generate by np, np.array.shape gives the size infor
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # for a single epoch, every training pair in mini_batch contributes to the training
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        # dnb and dnw has been summed for len(mini_batch) times, normlization needed
        self.weights = [w - (eta*nw)/len(mini_batch) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta*nb)/len(mini_batch) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # return a tuple (nabla_b,nabla_w) representing the gradient for the cost function C_x.
        # nabla_w and nabla_b are layer by layer lists of numpy arrays, similar to self.biases and self.weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activaitons, layer by layer
        zs = []  # list to store all the z vectos, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activation[-2].tranpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transoise())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # given the current weights and biases, evaluate the correction rate
        # reture the numebr of test inputs for which the neual network outputs the corret result.
        # The neural network's output is assumed to be the index of whichever neuron in the final
        # layer has the highes activation

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # return vector of partial derivatives partial C_x/partial a for the output activation
        return (output_activations-y)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)/(1-sigmoid(z))
