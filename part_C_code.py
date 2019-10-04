# References:
# https://sudeepraja.github.io/Neural/
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
# https://cs.stackexchange.com/questions/31874/how-does-the-momentum-term-for-backpropagation-algorithm-work
# https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf
# https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
# https://codesachin.wordpress.com/2015/12/06/backpropagation-for-dummies/

import numpy as np
import scipy.special
import matplotlib.pyplot as plt


# neuralNetwork class
class neuralNetwork:

    # initialise the neuralNetwork
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, momentum):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        self.wih = np.random.normal(0.0, pow(self.inodes + 1, -0.5), (self.hnodes, self.inodes + 1))
        self.who = np.random.normal(0.0, pow(self.hnodes + 1, -0.5), (self.onodes, self.hnodes + 1))

        self.delta_wih_old = []
        self.delta_who_old = []

        # learning rate
        self.lr = learningrate

        # momentum
        self.momentum = momentum

        # sigmoid activation function
        self.activation = lambda x: (1 + np.exp(-x))**-1

        # gradient of sigmoid activation function
        self.gradient = lambda x: x * (1 - x)

        # previous delta (momentum helps in jumping out from the valley and converge faster)
        self.previous_delta = lambda x: 0 if len(x) == 0 else momentum * x[-1]

        self.errors = []
        pass

    def previous(self, x):

        if len(x)==0:
            return 0
        else:
            momentum = np.zeros(np.shape(x[0]))
            for i in range(len(x)):
                momentum = momentum + (self.momentum**(i+1))*x[-(i+1)]
            return momentum

    # train the neural network
    def train(self, inputs_list, targets_list):
        # add bias and convert inputs list to 2d array
        inputs = np.insert(np.array(inputs_list, ndmin=2).T, 0, 1, axis=0)
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)

        # add bias
        hidden_outputs = np.insert(hidden_outputs, 0, 1, axis=0)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        # final_outputs = self.activation(final_inputs)
        final_outputs = final_inputs
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        self.errors.append(np.average(output_errors**2))
        print("Errors: {}".format(output_errors))

        # calculate gradient at output layer
        # del2 = output_errors * self.gradient(final_outputs)
        del2 = output_errors

        # calculate gradient at hidden layer
        del1 = np.dot(self.who.T, del2) * self.gradient(hidden_outputs)

        # add momentum to delta
        delta_who = self.lr * np.dot(del2, np.transpose(hidden_outputs)) + self.previous_delta(self.delta_who_old)

        # update momentum
        self.delta_who_old.append(delta_who)

        # update the weights for the links between the hidden and output layers
        self.who += delta_who

        # remove bias from del1
        del1 = np.delete(del1, 0, 0)

        # add momentum to delta
        delta_wih = self.lr * np.dot(del1, np.transpose(inputs)) + self.previous_delta(self.delta_wih_old)

        # Update momentum
        self.delta_wih_old.append(delta_wih)

        # update the weights for the links between the input and hidden layers
        self.wih += delta_wih

        pass

    # query the neural network
    def query(self, inputs_list):
        # add bias and convert inputs list to 2d array
        inputs = np.insert(np.array(inputs_list, ndmin=2).T, 0, 1, axis=0)

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)

        # add bias
        hidden_outputs = np.insert(hidden_outputs, 0, 1, axis=0)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        # final_outputs = self.activation(final_inputs)
        final_outputs = final_inputs

        return final_outputs


# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 7
output_nodes = 1

# learning rate is 0.01
learning_rate = 0.01
momentum = 0.9
epochs = 50000

def generate():
    inputs = np.random.uniform(-2, 2, (80, 2))
    targets = []

    for input in inputs:
        targets.append([0.75*(input[0]**2) - 1.25*(input[1]**2)])
    return inputs, targets

inputs, targets = generate()
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, momentum)
for i in range(epochs):
    print("Iteration: {}".format(i+1))
    # for j in range(len(inputs)):
    #     nn.train(inputs[j], targets[j])
    nn.train(inputs, targets)

plt.plot(nn.errors)
plt.show()

def test():
    test_set = np.random.uniform(-2, 2, (100, 2))
    mse = []
    for i in range(100):
        expected = 0.75*(test_set[i][0]**2) - 1.25*(test_set[i][1]**2)
        actual = nn.query(test_set[i])
        error = (expected-actual[0][0])**2
        mse.append(error)
    print("MSE: {}".format(mse))
    print("MSE: {}".format(np.mean(mse)))
    print("STD MSE: {}".format(np.std(mse)))


print("Function Approximation: {}".format(nn.query([0, 0])))
print("Function Approximation: {}".format(nn.query([1, 0])))
print("Function Approximation: {}".format(nn.query([0, 1])))
print("Function Approximation: {}".format(nn.query([1, 1])))

test()