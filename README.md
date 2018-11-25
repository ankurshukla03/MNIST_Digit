# MNIST_Digit
- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. 
- You can find dataset and more information about this [here](https://www.kaggle.com/c/digit-recognizer).

# Model - MLP
- Predicted the Output using simple MLP(Multilayer perceptron) model. Used the 90-10% dataset for splitting into training and validation set. Check the code [here](https://github.com/ankurshukla03/MNIST_Digit/blob/master/Digit_MLP.ipynb). Got the accuracy of 98% from 95% when increased the number of epochs from 5 to 10 on validation dataset.
- Predicted Output on test dataset got 97% accuracy when submitted on kaggle.

# Model 1- CNN
- Using a Convolutional Neural Network for increasing the accuracy from 98% to 99% on validation set. This model only had one convulational layer.
- Got 98.5% accuray on test set when submitted on kaggle
Check the notebook [file](https://github.com/ankurshukla03/MNIST_Digit/blob/master/Notebooks/Digit_MLP.ipynb) for more information.

# Model 2 - CNN
- Using 4 convolutional layer in CNN. Added batch normalization for better results in our second model.
- Used Data Augmentation technique for generating more data.
- Got 99.514% accuracy for prediction on test when submitted on Kaggle 

# Model 3 - CNN 
- Using Emsemble of 10 CNNs
- Got 99.7% accuracy when submitted on kaggle
- Check the code [here](https://github.com/ankurshukla03/MNIST_Digit/blob/master/Notebooks/Ensemble_CNN.ipynb)

# Model 4 - FashionMNIST
- Applied Model3 model on Fashion MNIST dataset. Check the dataset [here](https://www.kaggle.com/zalando-research/fashionmnist)
- Got 92% accuracy when evaluated against test data
- Check the code [here](https://github.com/ankurshukla03/MNIST_Digit/blob/master/Notebooks/Ensemble_FMNIST.ipynb)

## Kaggle
- Model 3 and model 4 are kernel on Kaggle. You can check [here](https://www.kaggle.com/ankurshukla03)
