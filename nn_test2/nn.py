import numpy as np
from random import shuffle
import string

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("starting")
class NeuralNetwork:
    
    def __init__(self, Inodes, Hnodes, Onodes, lr):
        self.Inodes = Inodes
        self.Hnodes = Hnodes
        self.Onodes = Onodes
        self.weights_ih = np.random.rand(Hnodes, Inodes) * 2 - 1
        self.weights_ho = np.random.rand(Onodes, Hnodes) * 2 - 1
        self.biases_ih = np.random.rand(Hnodes, 1) * 2 - 1
        self.biases_ho = np.random.rand(Onodes, 1) * 2 - 1
        self.lr = lr #learning rate
        #self.weights_ih = np.ones((Hnodes, Inodes))
        #self.weights_ho = np.ones((Onodes, Hnodes))
        #self.biases_ih = np.ones((Hnodes, 1))
        #self.biases_ho = np.ones((Onodes, 1))
        print(self.weights_ih)
        print(self.weights_ho)
        print(self.biases_ih)
        print(self.biases_ho)
    
    
    def feedForward(self, inputs):
        #print("feeding forward")
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        elif isinstance(inputs, np.ndarray):
            pass
        else:
            return
        #print(np.reshape(inputs, (self.Inodes 1)))
        inputs = np.reshape(inputs, (self.Inodes, 1))
        #print(inputs)
        if inputs.size != self.Inodes:
            print("inconsistent input length")
            return
        
        hiddens = sigmoid(np.matmul(self.weights_ih, inputs) + self.biases_ih)
        #print(hiddens)
        outputs = sigmoid(np.matmul(self.weights_ho, hiddens) + self.biases_ho)
        return outputs, hiddens
    
    
    def backProp(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        inputs = np.reshape(inputs, (self.Inodes, 1))
        targets = np.reshape(targets, (self.Onodes, 1))
        #print(targets)
        ys, hiddens = self.feedForward(inputs)
        output_errors = ys - targets
        #print(ys)
        #print(errors)
        hidden_errors = np.matmul(np.transpose(self.weights_ho), output_errors)
        weight_deltas_o = -self.lr * (output_errors * ys * (1 - ys)) * np.transpose(hiddens)
        weight_deltas_h = -self.lr * (hidden_errors * hiddens * (1 - hiddens)) * np.transpose(inputs)
        bias_deltas_o = -self.lr * (output_errors * ys * (1 - ys)) * 1
        bias_deltas_h = -self.lr * (hidden_errors * hiddens * (1 - hiddens)) * 1
        
        #print(weight_deltas_o)
        #print(weight_deltas_h)
        #print(bias_deltas_o)
        #print(bias_deltas_h)
        return weight_deltas_o, weight_deltas_h, bias_deltas_o, bias_deltas_h
    
    def train(self, inputs, targets):
        
        weight_deltas_o, weight_deltas_h, bias_deltas_o, bias_deltas_h = self.backProp(inputs, targets)
        
        self.weights_ho += weight_deltas_o
        self.weights_ih += weight_deltas_h
        self.biases_ho += bias_deltas_o
        self.biases_ih += bias_deltas_h
    
    def check(self, inputs):
        #print(inputs)
        outputs, hiddens = self.feedForward(inputs)
        return outputs
        
    
#######################################

def convertToCheckNum(item):
    if len(item) < nn.Inodes:
        missing = nn.Inodes - len(item)
        color_string = item.lower() + ('_' * missing)
        numbers = [letter_dict[char] for char in color_string]
        #print(numbers)
        return numbers
    elif len(item[0]) > nn.Inodes:
        print("an item has too many letters!")
        return
"""
nn = NeuralNetwork(10, 30, 1, .05)



letter_dict = dict(zip(string.ascii_lowercase, [x / 27 for x in range(2, 28)]))
letter_dict[' '] = 1 / 27
letter_dict['_'] = 0 / 27
print(letter_dict)
"""


letter_dict = dict(zip(string.ascii_lowercase, [x / 28 for x in range(26)]))
letter_dict[' '] = 26 / 28
letter_dict['-'] = 27 / 28
letter_dict['_'] = 28 / 28

data = []
for item in raw_data:
    convertToCheckNum()
    # if len(item[0]) < nn.Inodes:
    #     missing = nn.Inodes - len(item[0])
    #     color_string = item[0].lower() + ('_' * missing)
    #     numbers = [letter_dict[char] for char in color_string]
    #     #print(numbers)
    #     data.append([numbers, item[1]])
    # elif len(item[0]) > nn.Inodes:
    #     print("an item has too many letters!")
    #     break
"""
#print(training_data)
nn = NeuralNetwork(2, 10, 1, .05)
data = [[[0, 0], 0],
        [[0, 1], 1],
        [[1, 0], 1],
        [[1, 1], 0]]



training_data = data * 2500
shuffle(training_data)
for i in range(len(training_data)):
    if i % 100 == 0:
        print("training:" + str(i))
    nn.train(training_data[i][0], training_data[i][1])

print(nn.check([0, 0]))
print(nn.check([0, 1]))
print(nn.check([1, 0]))
print(nn.check([1, 1]))

"""
print(convertToCheckNum('blue'))
print("blue percentage:" + str(nn.check(convertToCheckNum('blue'))))
print("green percentage:" + str(nn.check(convertToCheckNum('green'))))
print("red percentage:" + str(nn.check(convertToCheckNum('red'))))
print("bee percentage:" + str(nn.check(convertToCheckNum('bee'))))
print("grue percentage:" + str(nn.check(convertToCheckNum('grue'))))
print("gred percentage:" + str(nn.check(convertToCheckNum('gred'))))
"""


