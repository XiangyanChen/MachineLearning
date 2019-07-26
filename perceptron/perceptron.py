"""
This is the implementation of Perceptron,
which is accessible in https://github.com/FlameCharmander/MachineLearning,
accomplished by FlameCharmander,
and my csdn blog is https://blog.csdn.net/tudaodiaozhale,
contact me via 13030880@qq.com.
"""
import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sign(self, value): #sign function
        return 1 if value >= 0 else -1

    def train(self, data_set, labels):  #used to train
        lr = 1    #learning rate
        data_set = np.array(data_set)
        n = data_set.shape[0]   #n rows
        m = data_set.shape[1]   #m cols
        weights = np.zeros(m)   #initialize weight
        bias = 0
        i = 0
        while i < n:
            if (labels[i] * self.sign(np.dot(weights, data_set[i]) + bias) == -1):   #it means this point is wrong
                weights = weights + lr * labels[i] * data_set[i] #update weights, this is corresponding to (2.6) in P28
                bias = bias + lr * labels[i] ##update bias, this is corresponding to (2.6)
                i = 0
            else:
                i += 1
        self.weights = weights
        self.bias = bias

    def predict(self, data):    #used to predict
        if (self.weights is not None and self.bias is not None):
            return self.sign(np.dot(self.weights, data) + self.bias)
        else:
            return 0


if __name__ == "__main__":
    """
        this code is corresponding to algorithm(2.1) in P29
    """
    data_set = [[3, 3],
               [4, 3],
               [1, 1]]
    labels = [1, 1, -1]
    perceptron = Perceptron()
    perceptron.train(data_set, labels)
    print("weights is:", perceptron.weights)
    print("bias is:", perceptron.bias)
    result = perceptron.predict([3, 3])
    print("prediction:", result)