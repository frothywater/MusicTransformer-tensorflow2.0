import os
import pickle

import tensorflow as tf

import params
from data import Data
from gen_utils import SampleStrategy
from model import MusicTransformerDecoder
from processor import decode_midi


def main():
    # load data
    dataset = Data(params.dataset_dir)
    test_files = dataset.file_dict["test"]
    os.makedirs(params.generated_dir, exist_ok=True)

    mt = MusicTransformerDecoder(
        embedding_dim=params.embedding_dim,
        vocab_size=params.vocab_size,
        num_layer=params.num_layer,
        max_seq=params.max_seq,
        dropout=params.dropout,
        loader_path=params.model_dir)

    device = f"/device:GPU:{params.gpu_id}"
    sample_strategy = SampleStrategy(temp=1.5, k=10)

    for i, file in enumerate(test_files):
        filename = os.path.basename(file).replace(".mid.pickle", "")
        print(f"[{i+1}/{len(test_files)}] {filename}")
        generated_path = os.path.join(params.generated_dir, f"{filename}_generated.mid")
        original_path = os.path.join(params.generated_dir, f"{filename}_original.mid")

        with open(file, "rb") as pickle_file:
            words = pickle.load(pickle_file)

        with tf.device(device):
            generated = mt.generate(sample_strategy, words)

        decode_midi(generated, file_path=generated_path)
        decode_midi(words, file_path=original_path)


if __name__ == "__main__":
    main()
