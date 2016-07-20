# deep-learning
Deep learning project in TensorFlow and Torch to analyze clinical health records and construct deep learning models to predict future patient complications.

## Background
This project uses **Stacked Denoising Autoencoders (SDA)** [[P. Vincent]](http://jmlr.csail.mit.edu/papers/volume11/vincent10a/vincent10a.pdf) to perform a feature representation on a given dataset. During unsupervised pre-training, parameters in the neural network are learned and configured layer by layer greedily by minimizing the reconstruction loss between each input and its decoded counterpart. A supervised softmax classifier on top of the network provides tuning for all parameters of the network (weights and biases for each layer).

## Usage
Not configured for general usage yet: currently reads train/test data from csv files in batch style and performs pre-training/tuning on a multi(4)-gpu system.

```python
# Start a TensorFlow session
sess = tf.Session()

# Initialize an unconfigured autoencoder with specified dimensions, etc.
sda = SDAutoencoder(dims=[784, 256, 64, 10],
                    activations=["sigmoid", "tanh", "sigmoid"],
                    sess=sess, 
                    noise=0.1,
                    loss="cross-entropy")

# Pretrain weights and biases of each layers in the network.
sda.pretrain_network(X_TRAIN_PATH)

# Read in test y-values to softmax classifier.
sda.finetune_parameters(X_TRAIN_PATH, Y_TRAIN_PATH, output_dim=2)

# Write to file the newly represented features.
sda.write_encoded_input("../data/transformed.csv", X_TEST_PATH)
```

## Current status
- (WIP) SDA implemented in final_sda.py in TensorFlow. Bugs present.
- (To do) Enable multi-gpu support in the architecture.
- (To do) Add compatibility for other data-loading methods
- (To do) Add pre-processing methods in TF
- (To do) More documentation
