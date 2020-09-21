import tensorflow as tf
import pandas as pd
import numpy as np
import metrics
import os
from tensorflow import keras

from sklearn import preprocessing
from pathlib import Path
from load_dataset import load_dataset
from social_lstm import SocialLSTM
from losses import compute_loss

def main():
    model = SocialLSTM(pred_len=12, cell_side=4, n_side_cells=9,
                       lstm_dim=32, emb_dim=32)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.003)
    model.compile(optimizer=optimizer, loss=compute_loss,
        metrics=[metrics.abe, metrics.fde])

    data = pd.read_csv('../../data/processed/trajectories.csv').rename({"particle": "id"}, axis=1)

    pct_idx = int(0.05 * len(data.index))

    # prepare datasets
    train_ds, n_train_samples = load_dataset(
        df=data[:pct_idx],
        obs_len=8,
        pred_len=12)

    # running on CPU because the batch size of 1 makes it worse when run on GPU
    with tf.device('/device:CPU:0'):
        model.run_eagerly = True
        # train the model on one epoch to load the variables
        model.fit(x=train_ds, epochs=1, steps_per_epoch=n_train_samples)

    model.load_weights('../../models/social_lstm/10.h5')
    model.summary()

    test_ds, n_test_samples = load_dataset(
        df=data[int(0.8 * len(data.index)):],
        obs_len=8,
        pred_len=12)

    # # running on CPU because the batch size of 1 makes it worse when run on GPU
    # with tf.device('/device:CPU:0'):
    #     model.run_eagerly = True
    #     # train the model on one epoch to load the variables
    #     model.fit(x=test_ds, epochs=1, steps_per_epoch=n_test_samples)

    results = model.evaluate(x=test_ds, verbose=1, steps=n_test_samples)
    print(results)


if __name__ == '__main__':
    main()