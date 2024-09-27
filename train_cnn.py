import numpy as np
import matplotlib.pyplot as plt
import pvml


def accuracy(net, X, Y):
    ''' Compute the accuracy.

    : param net: MLP neural network.
    : param X: array like.
    : param Y: array like.
    : return acc * 100: number.
    
    '''
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100



X_train = np.load('kmnist-train-imgs.npz')['arr_0']
Ytrain = np.load('kmnist-train-labels.npz')['arr_0']
X_test = np.load('kmnist-test-imgs.npz')['arr_0']
Ytest = np.load('kmnist-test-labels.npz')['arr_0']

#Now the arrays are 60.000 images 28x28. We want it to be 60.000 x 784.

Xtrain = X_train.reshape(X_train.shape[0], -1)
Xtest = X_test.reshape(X_test.shape[0], -1)

#Xtrain = np.file("kminst-train-imgs.npz")
#Ytrain = np.file("kmnist-train-labels.npz")
#Xtest = "kmnist-test-imgs.npz"
#Ytest = "kmnist-test-labels.npz"

print(X_train.shape, X_test.shape)
print("Training set after first reshape: ", Xtrain.shape,Ytrain.shape)
print("Test set after second reshape: ", Xtest.shape, Ytest.shape)


#Xtrain, Ytrain = np.load("train.npz").values()
Xtrain = Xtrain.reshape(60000, 28, 28, 1) - 0.5
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
#Xtest, Ytest = np.load("test.npz").values()
Xtest = Xtest.reshape(10000, 28, 28, 1) - 0.5
print("Test set after reshape: ", Xtest.shape, Ytest.shape)


network = pvml.CNN([1, 12, 32, 48, 10], [7, 3, 3, 3], [2, 2, 1, 1], [0, 0, 0, 0])

epochs = []
batch_size = 100
lr = 0.0001

train_accs = []
test_accs = []

print('Training: starting')
plt.ion()
for epoch in range(51):
    steps = Xtrain.shape[0] // batch_size
    network.train(Xtrain, Ytrain, lr=lr, steps=steps, batch=batch_size)
    train_acc = accuracy(network, Xtrain, Ytrain)
    test_acc = accuracy(network, Xtest, Ytest)
    print("Epoch number", epoch, "  Training Accuracy:", train_acc, "Test accuracy:", test_acc)

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    epochs.append(epoch)

    plt.clf()
    plt.plot(epochs, train_accs, color = 'r')
    plt.plot(epochs, test_accs, color ='blue')
    plt.title('Convolutional Neural Network accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.legend(["train", "test"])
    plt.pause(0.01)
    plt.savefig('CNN.png')
    network.save("CNN.npz")

plt.savefig('CNN.png')
network.save('CNN.npz')
plt.ioff()
print('Training: finished')
plt.show()

### evaluation
train_predictions, train_probs = network.inference(Xtrain)
train_acc = 100 * (train_predictions == Ytrain).mean()
print('training accuracy: ', train_acc)

test_predictions, test_probs = network.inference(Xtest)
test_acc = 100 * (test_predictions == Ytest).mean()
print('test accuracy: ', test_acc)




  
