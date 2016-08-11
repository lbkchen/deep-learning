# deep-learning
Deep learning project in TensorFlow and Torch to analyze clinical health records and construct deep learning models to predict future patient complications.

## Background
This project uses **Stacked Denoising Autoencoders (SDA)** [[P. Vincent]](http://jmlr.csail.mit.edu/papers/volume11/vincent10a/vincent10a.pdf) to perform feature learning on a given dataset. Two overall steps are necessary for fully configuring the network to encode the input data: **pre-training**, and **fine-tuning**.

During unsupervised pre-training, parameters in the neural network are learned and configured layer by layer greedily by minimizing the reconstruction loss between each input and its decoded counterpart. A supervised softmax classifier on top of the network provides fine tuning for all parameters of the network (weights and biases for each autoencoder layer plus softmax weights/biases).

Following this configuration, the input data can be read into the model and encoded into a different representation depending on the user's desired parameters (layer dimensions, activations, noise level, etc.). For example, this technique can be used to transform a sparse feature space of 30000 dimensions into a dense feature space of 400 dimensions as a primer for better training performance.

## Usage
The current working source code is located in `tf/sdautoencoder.py`. Currently reads train/test data from csv files in batch style. The following three datasets must be present for the SDA to output newly learned features:
- X training values
- Y training values
- X testing values

An additional dataset is needed if the output of SDA encoding is directly used for classification via the provided softmax classifier:
- Y testing values


In the future, a version of the program will be constructed to be optimized on a multi(4)-gpu system.

```python
# Start a TensorFlow session
sess = tf.Session()

# Initialize an unconfigured autoencoder with specified dimensions, etc.
sda = SDAutoencoder(dims=[784, 256, 64, 32],
                    activations=["sigmoid", "tanh", "sigmoid"],
                    sess=sess,
                    noise=0.1,
                    loss="rmse")

# Pretrain weights and biases of each layer in the network.
sda.pretrain_network(X_TRAIN_PATH)

# Read in test y-values to softmax classifier.
sda.finetune_parameters(X_TRAIN_PATH, Y_TRAIN_PATH, output_dim=10)

# Write to file the newly represented features.
sda.write_encoded_input(filepath="../data/transformed.csv", X_TEST_PATH)
```

For an example of how training is performed and subsequent accuracy is evaluated, a basic procedure is implemented on the MNIST data set in `tf/mnist_sda.py`.

## Performance
Testing on the MNIST data set, the softmax classifier on top of features extracted from the deep feature learning of the SDA can achieve approximately **98.3%** accuracy in identifying the digits. To achieve this result, the model in `tf/mnist_sda.py` is set up with the following parameters (which may not necessarily be optimal) with 500000 data points for layer-wise pretraining and 3000000 data points for fine tuning:

```python
sda = SDAutoencoder(dims=[784, 400, 200, 80],
                    activations=["sigmoid", "sigmoid", "sigmoid"],
                    sess=sess,
                    noise=0.20,
                    loss="cross-entropy",
                    pretrain_lr=0.0001,
                    finetune_lr=0.0001)
```
Total execution time for feature learning, training, and evaluation was just under 9 minutes on my 1.3 GHz MacAir processor (under a minute on a GPU machine using one GTX 1080). This result improves upon the benchmark of 92% achieved by just a [simple softmax classifier](https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners) without feature learning. It is also comparable to some simple 2D convolutional network models, which are optimized to take advantage of the 2D structures in image data.

In the future, we plan to do additional testing to optimize hyperparameters in the model and improve execution speed in various parts of the model.

## Current status
- (Done) SDA implemented in final_sda.py in TensorFlow.
- (Done) Implement softmax classifier.
- (To do) Implement command line execution of program.
- (WIP) Testing for any silent bugs.
- (To do) Enable multi-gpu support in the architecture.
- (WIP) Add compatibility for other data-loading methods
- (To do) Add pre-processing methods in TF
- (WIP) More documentation
