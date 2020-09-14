import tensorflow as tf
import pandas as pd
import numpy as np
import metrics
import os

from sklearn import preprocessing
from pathlib import Path
from load_dataset import load_dataset
from social_lstm import SocialLSTM
from losses import compute_loss


def main():
    # TODO: pass load arguments via config file

    model = SocialLSTM(pred_len=12, cell_side=4, n_side_cells=9,
                       lstm_dim=32, emb_dim=16)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.003)

    data = pd.read_csv('../../data/processed/trajectories.csv').rename({"particle": "id"}, axis=1)

    min_max_scaler_x = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()

    data_scaled = data.copy()

    data_scaled['x'] = min_max_scaler_x.fit_transform(np.reshape(data['x'].values.astype(float), (-1, 1)))
    data_scaled['y'] = min_max_scaler_y.fit_transform(np.reshape(data['y'].values.astype(float), (-1, 1)))

    pct_idx = int(0.8 * len(data.index))

    # prepare datasets
    train_ds, n_train_samples = load_dataset(
        df=data_scaled[:pct_idx],
        obs_len=8,
        pred_len=12)

    test_ds, n_test_samples = load_dataset(
        df=data_scaled[pct_idx:],
        obs_len=8,
        pred_len=12)

    out_dir = '../../models/social_lstm'

    def save_weights_func(epoch, _):
        model.save_weights(Path(out_dir, f'{epoch + 1:02d}.h5').as_posix())

    save_weights_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=save_weights_func,
        on_epoch_end=save_weights_func)

    model.compile(optimizer=optimizer, loss=compute_loss,
        metrics=[metrics.abe, metrics.fde])

    # running on CPU because the batch size of 1 makes it worse when run on GPU
    with tf.device('/device:CPU:0'):
        model.run_eagerly = True
        model.fit(x=train_ds, epochs=10, steps_per_epoch=n_train_samples,
            validation_data=test_ds, validation_steps=n_test_samples,
            callbacks=[save_weights_callback])


if __name__ == '__main__':
    main()
