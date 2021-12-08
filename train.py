from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import sys
import time
import argparse
import datetime

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, help='학습률', type=float)
parser.add_argument('--batch_size', default=2, help='batch size', type=int)
parser.add_argument('--pickle_dir', default='music', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--epochs', default=100, help='에폭 수', type=int)
parser.add_argument('--load_path', default=None, help='모델 로드 경로', type=str)
parser.add_argument('--save_path', default="result/dec0722", help='모델 저장 경로')
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=True)
parser.add_argument('--num_layers', default=6, type=int)

args = parser.parse_args()


# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu
num_layer = args.num_layers


# load data
dataset = Data('data/dataset')
print(dataset)


# load model
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# define model
mt = MusicTransformerDecoder(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)


# define tensorboard writer
train_log_dir = 'logs/'+par.train_id+'/train'
eval_log_dir = 'logs/'+par.train_id+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)


# Train Start
print(">>> Start training")
step = 0
batch_count = len(dataset.files) // batch_size

for e in range(epochs):
    mt.reset_metrics()
    epoch_start_time = time.time()

    for b in range(batch_count):
        batch_start_time = time.time()
        batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
        result_metrics = mt.train_on_batch(batch_x, batch_y)
        batch_end_time = time.time()

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', result_metrics[0], step=step)
            tf.summary.scalar('accuracy', result_metrics[1], step=step)

        batch_time = batch_end_time - batch_start_time
        sys.stdout.write(
            f"epoch: {e+1}/{epochs}, batch: {b+1}/{batch_count} | "
            + f"loss: {result_metrics[0]:5f}, accuracy: {result_metrics[1]:5f}, time: {batch_time:3f}s\r"
        )
        sys.stdout.flush()
        step += 1

    # evaluate
    eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
    eval_result_metrics, _ = mt.evaluate(eval_x, eval_y)
    mt.save(save_path)
    epoch_end_time = time.time()

    with eval_summary_writer.as_default():
        mt.sanity_check(eval_x, eval_y, step=e)
        tf.summary.scalar('loss', eval_result_metrics[0], step=step)
        tf.summary.scalar('accuracy', eval_result_metrics[1], step=step)

    epoch_time = epoch_end_time - epoch_start_time
    print(
        f"Epoch: {e+1}/{epochs}, Eval Loss: {eval_result_metrics[0]:5f}, "
        + f"Eval Accuracy: {eval_result_metrics[1]:5f}, Time: {epoch_time:3f}s"
    )
