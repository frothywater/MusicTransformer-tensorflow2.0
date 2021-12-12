import os
import sys
import time

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

import params
from custom import callback
from data import Data
from model import MusicTransformerDecoder


def main():
    device = f"/device:gpu:{params.gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
    tf.executing_eagerly()

    # load data
    dataset = Data(params.dataset_dir)
    print(dataset)

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
            loader_path=params.load_dir)
        mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)

    # define tensorboard writer
    train_log_dir = 'logs/' + params.train_id + '/train'
    eval_log_dir = 'logs/' + params.train_id + '/eval'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

    # Train Start
    print(">>> Start training")
    step = 0
    batch_count = len(dataset.file_dict["train"]) // params.batch_size

    for e in range(params.epochs):
        mt.reset_metrics()
        epoch_start_time = time.time()

        for b in range(batch_count):
            batch_start_time = time.time()
            batch_x, batch_y = dataset.slide_seq2seq_batch(params.batch_size, params.max_seq)
            with tf.device(device):
                result_metrics = mt.train_on_batch(batch_x, batch_y)
            batch_end_time = time.time()

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', result_metrics[0], step=step)
                tf.summary.scalar('accuracy', result_metrics[1], step=step)

            batch_time = batch_end_time - batch_start_time
            sys.stdout.write(
                f"epoch: {e+1}/{params.epochs}, batch: {b+1}/{batch_count} | "
                + f"loss: {result_metrics[0]:5f}, accuracy: {result_metrics[1]:5f}, time: {batch_time:3f}s\r"
            )
            sys.stdout.flush()
            step += 1

        # evaluate
        eval_x, eval_y = dataset.slide_seq2seq_batch(params.batch_size, params.max_seq, 'eval')
        with tf.device(device):
            eval_result_metrics, _ = mt.evaluate(eval_x, eval_y)

        epoch_end_time = time.time()

        if (e + 1) % 10 == 0:
            mt.save(params.model_dir, epoch=e+1)

        with eval_summary_writer.as_default():
            mt.sanity_check(eval_x, eval_y, step=e)
            tf.summary.scalar('loss', eval_result_metrics[0], step=step)
            tf.summary.scalar('accuracy', eval_result_metrics[1], step=step)

        epoch_time = epoch_end_time - epoch_start_time
        print(
            f"Epoch: {e+1}/{params.epochs}, Eval Loss: {eval_result_metrics[0]:5f}, "
            + f"Eval Accuracy: {eval_result_metrics[1]:5f}, Time: {epoch_time:3f}s"
        )


if __name__ == "__main__":
    main()
