import tensorflow as tf
import numpy as np
from .deep_sda import SDAutoencoder
from sklearn.preprocessing import StandardScaler

# train = np.genfromtxt("data/SAMX.csv", delimiter=",")
# test = np.genfromtxt("data/SAMY.csv", delimiter=",")

def main():
    data = np.genfromtxt("data/sSAMTablePart01.csv", delimiter=",")
    print(len(data), data.shape)

if __name__ == "__main__":
    main()