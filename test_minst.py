import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from NeuralNetwork.Dense.dense import Dense
from Functions.activations import Tanh
from Functions.losses import mse, mse_prime
from NeuralNetwork.network import train,predict

def preprocess(x,y,limit):
    x = x.reshape(x.shape[0], 28*28, 1)
    x = x.astype("float32") / 255

    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)

    return x[:limit], y[:limit]


(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train, y_train = preprocess(x_train, y_train, 2000)
x_test, y_test = preprocess(x_test, y_test, 100)

network = [
    Dense(28*28 , 32),
    Tanh(),
    Dense(32, 10),
    Tanh()
]

# train
train(network, mse, mse_prime, x_train, y_train, epochs=200, learning_rate=0.1)

# test
with open('log.txt','a') as f:
    test_results=[]

    for x,y in zip(x_test,y_test):
        output = predict(network,x)
        test_results.append((np.argmax(output),np.argmax(y)))
    correct=0
    for result in test_results:
        line = 'prediction: '+ str(result[0])+' actual: '+ str(result[1])
        print(line)
        f.write(line + '\n')
        if result[0] == result[1]:
            correct+=1
    line = "Accuracy of model on test input is: " + str((correct/len(test_results))*100)
    print(line)
    f.write(line + '\n')
