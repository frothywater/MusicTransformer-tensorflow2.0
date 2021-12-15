import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

import params
from custom import callback
from model import MusicTransformerDecoder


def main():
    device = f"/device:gpu:{params.gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
    tf.executing_eagerly()

    # load data
    train_data = np.load(os.path.join(params.dataset_dir, "train.npz"))
    valid_data = np.load(os.path.join(params.dataset_dir, "valid.npz"))
    train_x = train_data["x"]
    train_y = train_data["y"]
    valid_x = valid_data["x"]
    valid_y = valid_data["y"]

    # load model
    learning_rate = callback.CustomSchedule(params.embedding_dim)
    opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # define model
    with tf.device(device):
        mt = MusicTransformerDecoder(
            embedding_dim=params.embedding_dim,
            vocab_size=params.vocab_size,
            num_layer=params.num_layer,
            max_seq=params.max_seq,
            dropout=params.dropout,
            loader_path=params.load_dir,
        )
        mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)

    # define tensorboard writer
    train_log_dir = "logs/" + params.train_id + "/train"
    valid_log_dir = "logs/" + params.train_id + "/valid"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    # Train Start
    print(">>> Start training")
    step = 0
    batch_count = len(train_x)
    valid_batch_count = len(valid_x)
    min_valid_loss = None
    times_valid_loss_increased = 0
    early_stop_patience = 5

    for e in range(params.epochs):
        mt.reset_metrics()
        epoch_start_time = time.time()

        # Train
        loss_list = []
        accuracy_list = []
        for b in range(batch_count):
            batch_start_time = time.time()
            with tf.device(device):
                result_metrics = mt.train_on_batch(train_x[b], train_y[b])
            batch_end_time = time.time()

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", result_metrics[0], step=step)
                tf.summary.scalar("accuracy", result_metrics[1], step=step)

            batch_time = batch_end_time - batch_start_time
            sys.stdout.write(
                f"epoch: {e+1}/{params.epochs}, batch: {b+1}/{batch_count} | "
                + f"loss: {result_metrics[0]:.5f}, accuracy: {result_metrics[1]:.5f}, time: {batch_time:.3f}s\r"
            )
            sys.stdout.flush()
            loss_list.append(result_metrics[0])
            accuracy_list.append(result_metrics[1])
            step += 1

        # Validate
        valid_loss_list = []
        valid_accuracy_list = []
        for b in range(valid_batch_count):
            batch_start_time = time.time()
            with tf.device(device):
                metrics, _ = mt.evaluate(valid_x[b], valid_y[b])
            batch_end_time = time.time()

            batch_time = batch_end_time - batch_start_time
            sys.stdout.write(
                f"valid: {e+1}/{params.epochs}, batch: {b+1}/{valid_batch_count} | "
                + f"loss: {metrics[0]:.5f}, accuracy: {metrics[1]:.5f}, time: {batch_time:.3f}s\r"
            )
            sys.stdout.flush()
            valid_loss_list.append(metrics[0])
            valid_accuracy_list.append(metrics[1])

        epoch_end_time = time.time()

        avg_loss = np.mean(loss_list)
        avg_accuracy = np.mean(accuracy_list)
        avg_valid_loss = np.mean(valid_loss_list)
        avg_valid_accuracy = np.mean(valid_accuracy_list)

        with valid_summary_writer.as_default():
            tf.summary.scalar("loss", avg_valid_loss, step=step)
            tf.summary.scalar("accuracy", avg_valid_accuracy, step=step)

        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch: {e+1}/{params.epochs}, Time: {epoch_time:.3f}s")
        print(f"\t Valid Loss: {avg_valid_loss:.5f}, Valid Accuracy: {avg_valid_accuracy:.5f}")
        print(f"\t Train Loss: {avg_loss:.5f}, Train Accuracy: {avg_accuracy:.5f}")

        mt.save(params.model_dir, epoch=e + 1)

        # Early stopping
        if min_valid_loss is None or avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            times_valid_loss_increased = 0
        else:
            times_valid_loss_increased += 1
        if times_valid_loss_increased >= early_stop_patience:
            print("Early stopped.")
            break

if __name__ == "__main__":
    main()
