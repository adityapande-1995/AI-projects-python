import numpy as np
import copy

# Simple numpy NN

class NeuralNetwork_1:
    def __init__(self,sizes,weights=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        if not weights:
            self.biases = [np.random.randn(y, 1)*2 for y in sizes[1:]]
            self.weights = [np.random.randn(y, x)*2 for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.biases, self.weights = copy.deepcopy(weights[0]), copy.deepcopy(weights[1])
    
    def predict(self,a):
        inp = np.reshape( np.array(a), (5,-1) )
        for b, w in zip(self.biases, self.weights): 
            inp = sigmoid(np.dot(w, inp)+b)
        return inp
    
    def deepcopy(self):
        return NeuralNetwork_1(self.sizes, weights = [self.biases, self.weights] )
    
    def mutate(self,q=0.1):
        for item in self.biases:
            row,col = item.shape
            for i in range(0,row):
                for j in range(0,col):
                    if np.random.random() < q : item[i,j] *= (np.random.random()*2 -1)*0.1 + 1

        for item in self.weights:
            row,col = item.shape
            for i in range(0,row):
                for j in range(0,col):
                    if np.random.random() < q : item[i,j] *= (np.random.random()*2 -1)*0.1 + 1

    def signature(self):
        s = 0
        for item in self.biases+self.weights:
            s += sum(item.flatten().tolist())

        return s
        


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


 # --------------- Tensorflow based NN ----------------#
import tensorflow as tf

class NeuralNetwork_2:
    def __init__(self,sizes,weights=None):
        self.sizes = sizes
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(sizes[0], input_dim=sizes[0], activation='relu'))
        self.model.add(tf.keras.layers.Dense(sizes[1], activation='relu'))
        self.model.add(tf.keras.layers.Dense(sizes[2], activation='sigmoid'))
        if weights :
            self.model.set_weights(weights)

    def predict(self,input):  # 'input' is a list
        temp = np.reshape(np.array(input), (-1,4)) # IMPORTANT LINE !!!
        return self.model.predict(temp)

    def deepcopy(self):
        return NeuralNetwork_2(self.sizes, weights = copy.deepcopy(self.model.get_weights()) )
    
    def mutate(self,q=0.1):
        wt_array = self.model.get_weights()
        for i in range(0,len(wt_array)):
            if i%2 == 0: # Even number, 2d array
                row,col = wt_array[i].shape
                temp = wt_array[i]
                for j in range(0,r):
                    for k in range(0,c):
                        temp[j,k] *= (np.random.random()*2 -1)*q + 1 
            else: # Odd number, 1d array
                l = wt_array[i].shape
                temp = wt_array[i]
                for j in range(0,l):
                    temp[j] *= (np.random.random()*2 -1)*q + 1
        
        self.model.set_weights(copy.deepcopy(wt_array))

    def signature(self):
        wt_array = self.model.get_weights()
        s = 0
        for item in wt_array:
            s += sum(item.flatten().tolist())
        
        return s 
        





        
