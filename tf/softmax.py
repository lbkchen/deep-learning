import tf.final_sda as sda
import tensorflow as tf
import numpy as np


@sda.stopwatch
def main():
    sess = tf.Session()
    # autoencoder = sda.SDAutoencoder(dims=[3997, 500, 500, 500],
    #                                 activations=["sigmoid", "sigmoid", "sigmoid"],
    #                                 sess=sess,
    #                                 noise=0.05,
    #                                 loss="rmse",
    #                                 print_step=50)
    gen = sda.get_batch_generator("../data/sSAMTablePart01.csv",
                                  20,
                                  repeat=500)
    index = 0
    for thing in gen:
        index += 1
        print(thing[0][0])

    print("Total", index)

if __name__ == "__main__":
    main()