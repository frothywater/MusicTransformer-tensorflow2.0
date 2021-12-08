import argparse
import os
import pickle

import tensorflow as tf

import params as par
from data import Data
from gen_utils import SampleStrategy
from model import MusicTransformerDecoder
from processor import decode_midi

parser = argparse.ArgumentParser()

parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--load_path', default=None, help='모델 로드 경로', type=str)
parser.add_argument('--save_path', default=None, type=str)
parser.add_argument('--pickle_dir', default=None, type=str)

args = parser.parse_args()


# set arguments
max_seq = args.max_seq
load_path = args.load_path
save_path = args.save_path
pickle_dir = args.pickle_dir


# load data
dataset = Data(pickle_dir)
test_files = dataset.file_dict["test"]
os.makedirs(save_path, exist_ok=True)

mt = MusicTransformerDecoder(
    embedding_dim=256,
    vocab_size=par.vocab_size,
    num_layer=6,
    max_seq=max_seq,
    debug=False,
    loader_path=load_path
)

device = f"/device:GPU:{par.gpu_id}"
sample_strategy = SampleStrategy(temp=1.5, k=10)

for i, file in enumerate(test_files):
    filename = os.path.basename(file).replace(".mid.pickle", "")
    print(f"[{i+1}/{len(test_files)}] {filename}")
    generated_path = os.path.join(save_path, f"{filename}_generated.mid")
    original_path = os.path.join(save_path, f"{filename}_original.mid")

    with open(file, "rb") as pickle_file:
        words = pickle.load(pickle_file)

    with tf.device(device):
        generated = mt.generate(sample_strategy, words)

    decode_midi(generated, file_path=generated_path)
    decode_midi(words, file_path=original_path)
