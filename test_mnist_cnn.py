import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from NeuralNetwork.Dense.dense import Dense
from NeuralNetwork.CNN.convolutional import Convolutional
from NeuralNetwork.CNN.reshape import Reshape
from Functions.activations import Sigmoid
from Functions.losses import binary_cross_entropy, binary_cross_entropy_prime
from NeuralNetwork.network import train, predict

def preprocess_data(x, y, limit):
    # zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index))
    # all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)


test_results=[]

for x,y in zip(x_test,y_test):
    output = predict(network,x)
    test_results.append((np.argmax(output),np.argmax(y)))
correct=0
for result in test_results:
    print('prediction:',result[0],'actual:',result[1])
    if result[0] == result[1]:
        correct+=1

print("Accuracy of model on test input is:",(correct/len(test_results))*100)