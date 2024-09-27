# KmnistAnalysis
KMNIST dataset analysis.

Machine learning application on KMNIST dataset: 
- Multilayer perceptron training and testing
- Convolutional Neural Networks training and testing
- Normalization/Reduction (L2 / PCA)
- K Nearest Neighbours Algorithm
- Autoencoders for Image Reconstruction and Noise cleaning

Both the dataset itself and the contents of this repo are licensed under a permissive CC BY-SA 4.0 license, except where specified within some benchmark
scripts. CC BY-SA 4.0 license requires attribution, and we would suggest to use the following attribution to the KMNIST dataset.
"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341

Info on Kuzushiji writing here (https://www.kaggle.com/code/aakashnain/kmnist-mnist-replacement) (https://naruhodo.weebly.com/blog/introduction-to-kuzushiji) (https://www.simonwenkel.com/notes/ai/datasets/vision/Kuzushiji-MNIST.html)


PVML Library (https://github.com/claudio-unipv/pvml)

For training (train_cnn.py, train_mlp.py) 

For data analysis (evaluation.py, evaluationPCA.py,  data_analysis.py, identification.py, identificationVisualize.py, accuracy.py )


In Keras Library there are: 

- examples of PCA and its visualization (effect on image and visualization of eigenvalues).
- CNN architecture and KNN method
- Autoencoder (inspired by https://www.tensorflow.org/tutorials/generative/autoencoder?hl=it) for Reconstruction of Images and Noise Cleaning
