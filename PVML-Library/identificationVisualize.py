import numpy as np
import matplotlib.pyplot as plt
import pvml
import os

#we also return scores to verify which are the result with bigger confidence found by the system
##def inference(X, w, b):
##    scores = X @ w + b
##    labels = (scores > 0).astype(int)
##    return labels, scores
def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

####X, Y = load_reshape("train.npz")
####print("Training set after reshape: ", X.shape, Y.shape)
##Xtestt, Ytestt = load_reshape("test.npz")
####print("Test set after reshape: ", Xtestt.shape, Ytestt.shape)
##
##X, Y = np.load("train.npz").values()
####Xtest, Ytest = np.load("test.npz").values()
##Xtest, Ytest = load_reshape("test.npz")


Xtrain = np.load('kmnist-train-imgs.npz')['arr_0']
Ytrain = np.load('kmnist-train-labels.npz')['arr_0']
Xtest = np.load('kmnist-test-imgs.npz')['arr_0']
Ytest = np.load('kmnist-test-labels.npz')['arr_0']

#Now the arrays are 60.000 images 28x28. We want it to be 60.000 x 784.

Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
Xtest = Xtest.reshape(Xtest.shape[0], -1)




neuralnet = "mlp(2hiddentest).npz"
net = pvml.MLP.load(neuralnet)

##w = net.weights[0]
##b = net.biases[0]
##predictions,scores = net.inference(Xtest)
##min = [99,99,99]
##max = [0,0,0]
##pos_ind = []
##neg_ind = []


numberofclasses = 10
words = ["O","KI","SU","TSU","NA","HA","MA","YA","NE","WO"]

predictions, probs = net.inference(Xtest)
acc = 100 * (predictions == Ytest).mean()
print("Test accuracy: ", acc)

classes = ["O","KI","SU","TSU","NA","HA","MA","YA","NE","WO"]
classes.sort()
List = []
Predictions = []
##for number in range(3):
## neuralnet = f"mlp({number}hidden).npz"
## net = pvml.MLP.load(neuralnet)
## predictions, probs = net.inference(Xtest)
## max = np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)
## for j in range(predictions.size):
##    if predictions[j] != Ytest[j]:
##            if all(x < probs[j,predictions[j]] for x in max[:,0]):
##                print(probs[j, predictions[j]])
##                max[2,0] = probs[j,predictions[j]]
##                max[2,1] = j
##                max[2,2] = predictions[j]
##                max = max[np.argsort(max[:,0]),:][::-1]
##                List.append(j)
##                Predictions.append(predictions[j])
##
## print(max)
## print(List)


##for number in range(1):
net = pvml.MLP.load(neuralnet)
predictions, probs = net.inference(Xtest)
max = np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)
for j in range(predictions.size):
    if predictions[j] != Ytest[j]:
            if all(x < probs[j,predictions[j]] for x in max[:,0]):
                print(probs[j, predictions[j]])
                max[2,0] = probs[j,predictions[j]]
                max[2,1] = j
                max[2,2] = predictions[j]
                max = max[np.argsort(max[:,0]),:][::-1]
                List.append(j)
                Predictions.append(predictions[j])

print(max)
print(List)


w = 16
h = 16
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 4

for i in range(1,17):
    j = List[i-1]
    h = Predictions[i-1]
    image = Xtest[j, :]
    fig.add_subplot(rows, columns, i)
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.title(words[Ytest[j]] + " predicted as " + words[h] )
    plt.imshow(image,cmap='gray')
plt.show()

#BestValues = [17,23,787, 382,1300,2022]
#Bestpredictions = [2,5,3,0,5,9]

w = 16
h = 16
fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 2

for i in range(1,7):
    j = BestValues[i-1]
    h = Bestpredictions[i-1]
    image = Xtest[j, :]
    fig.add_subplot(rows, columns, i)
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.title(words[Ytest[j]] + " predicted as " + words[h] )
    plt.imshow(image,cmap='gray')
plt.show()            
    # specifichiamo le predizioni sbagliate
    
##for m in max:
##     files = os.listdir("images/test/" +classes[int(m[1]//20)] )
##     print("images/test/" + classes[int(m[1]//20)]+"/"+files[int(m[1]%20)])
##     print("Ind folder: ", classes[int(m[1]//20)], " Ind file: ", files[int(m[1]%20)], " Wrongly predicted class: ", classes[int(m[2])])
##    

















##for j in range(scores.size):
##    if predictions[j] != Y[j]:
##        if predictions[j] == 0:
##             if all(x > scores[j] for x in min):
##                min[2] = scores[j]
##                min.sort()
##                pos_ind.insert(0, j)
##        if predictions[j] == 1:
##            if all(x < scores[j] for x in max):
##                max[2] = scores[j]
##                max.sort(reverse=True)
##                neg_ind.insert(0, j - 6250)
##
##
##
##
##number_rev = 3 #number of reviews printed
##
##print("Worse predictions on the test set for negative values:")
##for file_index in pos_ind[:number_rev]:
##    print("Score: ", scores[file_index], "\n")
##    files = os.listdir("aclImdb/test/pos")
##    f = open("aclImdb/test/pos/" + files[file_index])
##    print(f.read())
##    f.close()
##    print("\n")
##
##print("Worse prediction on the test set for positive values")
##for file_index in neg_ind[:number_rev]:
##    print("Score: ", scores[file_index + 6250], "\n")
##    files = os.listdir("aclImdb/test/neg")
##    f = open("aclImdb/test/neg/" + files[file_index])
##    print(f.read())
##    f.close()
##    print("\n")
