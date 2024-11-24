import numpy as np
import matplotlib.pyplot as plt
import time
import random
import string
import copy

# t0 = time.process_time()

# i made this several years ago, and remember the network itself functioning properly but the autoencoder didn't, and i don't think i ever definitely determined why

class Activations:
    @staticmethod
    def sigmoid(x, derivative = False):
        if derivative:
            return Activations.sigmoid_der(x)
        else:
            return 1 / (1 + np.exp(-x))
    
    def sigmoid_der(x):
        return Activations.sigmoid(x) * (1 - Activations.sigmoid(x))
    
    def linear(x):
        return x

print("starting")

# np.random.seed(0)

class Layer:
    
    def __init__(self, in_nodes, out_nodes, activation_func, params=None):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation_func = activation_func
        if params is None:
            #self.weights = np.array([[-1, .5, .2], [.3, 1, -.4]])
            self.weights = np.random.rand(out_nodes, in_nodes) * 2 - 1
            #self.biases = np.array([.4, -1]).reshape((out_nodes), 1)
            self.biases = np.random.rand(out_nodes, 1) * 2 - 1
        else:
            self.weights = params[0]
            self.biases = params[1]
    
    def feed(self, inputs):
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        elif isinstance(inputs, np.ndarray):
            pass
        else:
            return
        inputs = np.array(inputs).reshape((len(inputs), 1))
        
        if self.activation_func == 'sigmoid':
            return Activations.sigmoid(np.matmul(self.weights, inputs) + self.biases)
        else:
            return 0
        # if self.activation_func == "none":
        #     return np.matmul(self.weights, inputs) + self.biases
        # elif self.activation_func == "sigmoid":
        #     return sigmoid(np.matmul(self.weights, inputs) + self.biases)
    
    def copyLayer(self):
        copyl = Layer(self.in_nodes, self.out_nodes, self.activation_func)
        copyl.weights = np.copy(self.weights)
        copyl.biases = np.copy(self.biases)
        return copyl

class NeuralNetwork2:
    
    def __init__(self, Inodes, Hlayers, Onodes, activation_func, lr, layer_params=None):
        self.Inodes = Inodes
        self.Hlayers = Hlayers # Hlayers is list where the length is number of layers, elems are # of nodes
        self.num_hiddens = len(Hlayers)
        self.num_layers = self.num_hiddens + 1
        self.Onodes = Onodes
        self.lr = lr
        self.activation_func = activation_func
        self.layers = []
        self.layer_outs = [0] * (self.num_hiddens + 1)
        if layer_params is None:
            self.layers.append(Layer(Inodes, Hlayers[0], activation_func))
            for i in range(self.num_hiddens - 1):
                self.layers.append(Layer(Hlayers[i], Hlayers[i + 1], activation_func))
            self.layers.append(Layer(Hlayers[-1], Onodes, activation_func))
        else:
            self.layers.append(Layer(Inodes, Hlayers[0], activation_func, params=layer_params[0]))
            for i in range(self.num_hiddens - 1):
                self.layers.append(Layer(Hlayers[i], Hlayers[i + 1], activation_func, params=layer_params[i + 1]))
            self.layers.append(Layer(Hlayers[-1], Onodes, activation_func, params=layer_params[-1]))
        # print(len(self.layers))
        # print(len(self.layer_outs))
        # print(self.layers)
        
    def copyNN(self):
        copynn = copy.deepcopy(self)
        return copynn
    
    def saveNNToFile(self, filename):
        c = self.copyNN()
        data = []
        for i in range(c.num_layers):
            data.append(c.layers[i].weights)
            data.append(c.layers[i].biases)
        data.append([c.Inodes, c.Hlayers, c.Onodes, c.activation_func, c.lr])
        np.save(filename, data)
    
    def loadNNFromFile(filename, lr=None): #lr=None means that the lr of the new network will be taken from the saved network
        dataloaded = np.load(filename, allow_pickle=True)
        nn_params = []
        for i in range(0, len(dataloaded[:-1]), 2):
            nn_params.append((dataloaded[i], dataloaded[i + 1]))
        new = NeuralNetwork2(dataloaded[-1][0], 
                             dataloaded[-1][1], 
                             dataloaded[-1][2], 
                             dataloaded[-1][3], 
                             (lr, dataloaded[-1][4])[lr is None], #this means "take the second element if lr is None, else use lr"
                             layer_params=nn_params)
        return new
    
    def setLR(self, lr):
        self.lr = lr
    
    def setOutDiffsFunction(self, output, target): # o_diffs means out_diffs
        return output - target
    
    def feedForward(self, inputs):
        self.layer_outs[0] = self.layers[0].feed(inputs)
        for i in range(self.num_hiddens - 1 + 1): # the +1 is for the output layer
            self.layer_outs[i + 1] = self.layers[i + 1].feed(self.layer_outs[i])
        return self.layer_outs
    
    def backProp(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        inputs = np.reshape(inputs, (self.Inodes, 1))
        targets = np.reshape(targets, (self.Onodes, 1))
        outs = self.feedForward(inputs)
        out_diffs = outs[-1] - targets
        self.output_errors = .5 * out_diffs * out_diffs
        self.output_errors_der = out_diffs # the derivative of output_errors
        hidden_errors = [0] * self.num_hiddens
        weight_deltas = [0] * (self.num_hiddens + 1)
        bias_deltas = [0] * (self.num_hiddens + 1)
        
        hidden_errors[-1] = np.matmul(self.layers[-1].weights.T, self.output_errors_der) * (outs[-2] * (1 - outs[-2]))
        for i in range(1, self.num_hiddens):
            hidden_errors[-i - 1] = np.matmul(self.layers[-i - 1].weights.T, hidden_errors[-i]) * (outs[-i - 2] * (1 - outs[-i - 2]))
        
        weight_deltas[-1] = -self.lr * np.matmul(self.output_errors_der * (outs[-1] * (1 - outs[-1])), outs[-2].T)
        bias_deltas[-1] = -self.lr * self.output_errors_der * (outs[-1] * (1 - outs[-1]))
        # for i in range(1, self.num_hiddens + 1):
        #     print(-i)
        #     print(hidden_errors[-i].shape)
        #     print(outs[-i].shape)
        
        for i in range(2, self.num_hiddens + 1):
            weight_deltas[-i] = -self.lr * np.matmul(hidden_errors[-i + 1], outs[-i - 1].T)
            bias_deltas[-i] = -self.lr * hidden_errors[-i + 1]
        weight_deltas[0] = -self.lr * np.matmul(hidden_errors[0], inputs.T)
        bias_deltas[0] = -self.lr * hidden_errors[0]
        
        return weight_deltas, bias_deltas
    
    def train(self, inputs, targets):
        
        weight_deltas, bias_deltas = self.backProp(inputs, targets)
        # print("---------------")
        # for i in range(self.num_hiddens + 1):
        #     print(i)
        #     print(weight_deltas[i].shape)
        #     print(self.layers[i].weights.shape)
        # print("===========================")
        for i in range(self.num_hiddens + 1):
            # print(i)
            # print(weight_deltas[i].shape)
            # print(self.layers[i].weights.shape)
            self.layers[i].weights += weight_deltas[i]
            self.layers[i].biases += bias_deltas[i]
        
    def check(self, inputs):
        #print(inputs)
        return self.feedForward(inputs)[-1]
    
    def getError(self):
        return np.sum(self.output_errors)
    
#######################################


class AutoEncoder:
    
    def __init__(self, Inodes, Hlayers, Onodes, activation_func, lr, layer_params=None):
        self.Inodes = Inodes
        self.Hlayers = Hlayers # Hlayers is list where the length is number of layers, elems are # of nodes
        self.num_hiddens = len(Hlayers)
        self.num_layers = self.num_hiddens + 1
        self.Onodes = Onodes
        self.lr = lr
        self.activation_func = activation_func
        self.layers = []
        self.layer_outs = [0] * (self.num_hiddens + 1)
        if layer_params is None:
            self.layers.append(Layer(Inodes, Hlayers[0], activation_func))
            for i in range(self.num_hiddens - 1):
                self.layers.append(Layer(Hlayers[i], Hlayers[i + 1], activation_func))
            self.layers.append(Layer(Hlayers[-1], Onodes, activation_func))
        else:
            self.layers.append(Layer(Inodes, Hlayers[0], activation_func, params=layer_params[0]))
            for i in range(self.num_hiddens - 1):
                self.layers.append(Layer(Hlayers[i], Hlayers[i + 1], activation_func, params=layer_params[i + 1]))
            self.layers.append(Layer(Hlayers[-1], Onodes, activation_func, params=layer_params[-1]))
        # print(len(self.layers))
        # print(len(self.layer_outs))
        # print(self.layers)
        
    def copyNN(self):
        copynn = copy.deepcopy(self)
        return copynn
    
    def saveNNToFile(self, filename):
        c = self.copyNN()
        data = []
        for i in range(c.num_layers):
            data.append(c.layers[i].weights)
            data.append(c.layers[i].biases)
        data.append([c.Inodes, c.Hlayers, c.Onodes, c.activation_func, c.lr])
        np.save(filename, data)
    
    def loadNNFromFile(filename, lr=None): #lr=None means that the lr of the new network will be taken from the saved network
        dataloaded = np.load(filename, allow_pickle=True)
        nn_params = []
        for i in range(0, len(dataloaded[:-1]), 2):
            nn_params.append((dataloaded[i], dataloaded[i + 1]))
        new = NeuralNetwork2(dataloaded[-1][0], 
                             dataloaded[-1][1], 
                             dataloaded[-1][2], 
                             dataloaded[-1][3], 
                             (lr, dataloaded[-1][4])[lr is None], #this means "take the second element if lr is None, else use lr"
                             layer_params=nn_params)
        return new
    
    def setLR(self, lr):
        self.lr = lr
    
    def setOutDiffsFunction(self, output, target): # o_diffs means out_diffs
        return output - target
    
    def feedForward(self, inputs, coder='both'):
        if coder == 'both':
            self.layer_outs[0] = self.layers[0].feed(inputs)
            for i in range(self.num_hiddens - 1 + 1): # the +1 is for the output layer
                self.layer_outs[i + 1] = self.layers[i + 1].feed(self.layer_outs[i])
            return self.layer_outs
        elif coder == 'encode':
            middle = (int)((self.num_hiddens) / 2)
            self.layer_outs[0] = self.layers[0].feed(inputs)
            for i in range(middle): # 
                self.layer_outs[i + 1] = self.layers[i + 1].feed(self.layer_outs[i])
            # print(self.layer_outs)
            return self.layer_outs[middle]
        elif coder == 'decode':
            middle = (int)((self.num_hiddens) / 2)
            self.layer_outs[middle + 1] = self.layers[middle + 1].feed(inputs)
            # print(self.layer_outs)
            for i in range(middle + 1, self.num_hiddens - 1 + 1): # 
                # print(self.layer_outs[i])
                self.layer_outs[i + 1] = self.layers[i + 1].feed(self.layer_outs[i])
            # print(self.layer_outs)
            return self.layer_outs[-1]
            
    
    def backProp(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        inputs = np.reshape(inputs, (self.Inodes, 1))
        targets = np.reshape(targets, (self.Onodes, 1))
        outs = self.feedForward(inputs)
        out_diffs = outs[-1] - targets
        self.output_errors = .5 * out_diffs * out_diffs
        self.output_errors_der = out_diffs # the derivative of output_errors
        hidden_errors = [0] * self.num_hiddens
        weight_deltas = [0] * (self.num_hiddens + 1)
        bias_deltas = [0] * (self.num_hiddens + 1)
        
        hidden_errors[-1] = np.matmul(self.layers[-1].weights.T, self.output_errors_der) * (outs[-2] * (1 - outs[-2]))
        for i in range(1, self.num_hiddens):
            hidden_errors[-i - 1] = np.matmul(self.layers[-i - 1].weights.T, hidden_errors[-i]) * (outs[-i - 2] * (1 - outs[-i - 2]))
        
        weight_deltas[-1] = -self.lr * np.matmul(self.output_errors_der * (outs[-1] * (1 - outs[-1])), outs[-2].T)
        bias_deltas[-1] = -self.lr * self.output_errors_der * (outs[-1] * (1 - outs[-1]))
        # for i in range(1, self.num_hiddens + 1):
        #     print(-i)
        #     print(hidden_errors[-i].shape)
        #     print(outs[-i].shape)
        
        for i in range(2, self.num_hiddens + 1):
            weight_deltas[-i] = -self.lr * np.matmul(hidden_errors[-i + 1], outs[-i - 1].T)
            bias_deltas[-i] = -self.lr * hidden_errors[-i + 1]
        weight_deltas[0] = -self.lr * np.matmul(hidden_errors[0], inputs.T)
        bias_deltas[0] = -self.lr * hidden_errors[0]
        
        return weight_deltas, bias_deltas
    
    def train(self, inputs, targets):
        
        weight_deltas, bias_deltas = self.backProp(inputs, targets)
        # print("---------------")
        # for i in range(self.num_hiddens + 1):
        #     print(i)
        #     print(weight_deltas[i].shape)
        #     print(self.layers[i].weights.shape)
        # print("===========================")
        for i in range(self.num_hiddens + 1):
            # print(i)
            # print(weight_deltas[i].shape)
            # print(self.layers[i].weights.shape)
            self.layers[i].weights += weight_deltas[i]
            self.layers[i].biases += bias_deltas[i]
    
    def encode(self, inputs):
        return self.feedForward(inputs, coder='encode')
        # middle = (int)((self.num_hiddens / 2) # this may not be thhe way to go about it
        # encoder = NeuralNetwork2(self.Inodes, self.Hlayers[:middle], self.Hlayers[middle], self.activation_func, self.lr, layer_params= )
    
    def decode(self, inputs):
        return self.feedForward(inputs, coder='decode')
    
    def check(self, inputs):
        #print(inputs)
        return self.feedForward(inputs)[-1]
    
    def getError(self):
        return np.sum(self.output_errors)

# nn2 = NeuralNetwork2(2, [5, 5], 1, Activations.sigmoid, .1)

# ae = AutoEncoder(10, [8, 5, 3, 5, 8], 10, 'sigmoid', .1)

# print(ae.encode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
# print(ae.decode([0, 1, 2]))

# data = [[[0, 0], 0],
#         [[0, 1], 1],
#         [[1, 0], 1],
#         [[1, 1], 0]]

# errors = []

# training_data = data * 2500
# random.shuffle(training_data)
# for i in range(len(training_data)):
#     nn2.train(training_data[i][0], training_data[i][1])
#     if i % 1000 == 0:
#         print("training:" + str(i))
#         errors.append(nn2.getError())    

# print(nn2.check([0, 0]))
# print(nn2.check([0, 1]))
# print(nn2.check([1, 0]))
# print(nn2.check([1, 1]))
# print("Total error:" + str(nn2.getError()))
# plt.plot(errors)
# plt.show()

# print("ELAPSED TIME: " + str(round(time.process_time() - t0, 2)) + " SECONDS")



