import numpy as np
import matplotlib.pyplot as plt

n = int(input("Which data you want to visualize? \n"))
n = int(n)
##n = 23

words = ["O","KI","SU","TSU","NA","HA","MA","YA","NE","WO"]

##Xtrain , Ytrain = np.load("train.npz").values()
##Xtest , Ytest = np.load("test.npz").values()

X_train = np.load('kmnist-train-imgs.npz')['arr_0']
Ytrain = np.load('kmnist-train-labels.npz')['arr_0']
X_test = np.load('kmnist-test-imgs.npz')['arr_0']
Ytest = np.load('kmnist-test-labels.npz')['arr_0']

#Now the arrays are 60.000 images 28x28. We want it to be 60.000 x 784.

Xtrain = X_train.reshape(X_train.shape[0], -1)
Xtest = X_test.reshape(X_test.shape[0], -1)



image = Xtest[n, :]
plt.imshow(image,cmap='gray')
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.title(words[Ytest[n]] + f"  (Data element: {n} , value = ({Ytest[n]}))" )
plt.colorbar()
plt.show()












