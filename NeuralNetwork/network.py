import numpy as np
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    with open('../log.txt','w') as f:
        for e in range(epochs):
            error = 0
            count = 0
            for x, y in zip(x_train, y_train):
                # forward
                count +=1
                output = predict(network, x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)
                if count % 10 == 0:
                    np.sum(output ==  y) / y.size

            error /= len(x_train)
            if verbose:
                line = f"{e + 1}/{epochs}, error={error}" 
                f.write(line)
                print(line)
            