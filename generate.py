import os
import pickle

import tensorflow as tf

import params
from generate_utils import SampleStrategy, bar_length, get_test_files
from model import MusicTransformerDecoder
from processor import decode_midi


def main():
    device = f"/device:gpu:{params.gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

    # load data
    test_files = get_test_files(params.words_dir)
    os.makedirs(params.generated_dir, exist_ok=True)

    with tf.device(device):
        mt = MusicTransformerDecoder(
            embedding_dim=params.embedding_dim,
            vocab_size=params.vocab_size,
            num_layer=params.num_layer,
            max_seq=params.max_seq,
            dropout=params.dropout,
            loader_path=params.model_dir,
            load_epoch=params.load_epoch)

    sample_strategy = SampleStrategy(temp=1.5, k=10)

    for i, file in enumerate(test_files):
        filename = os.path.basename(file).replace(".mid.pickle", "")
        print(f"[{i+1}/{len(test_files)}] {filename}")

        with open(file, "rb") as pickle_file:
            words = pickle.load(pickle_file)
        if bar_length(words) < 32:
            print("This song is shorter than 32 bars, skip.")
            continue

        generated_path = os.path.join(params.generated_dir, f"{filename}_generated.mid")
        original_path = os.path.join(params.generated_dir, f"{filename}_original.mid")

        with tf.device(device):
            generated = mt.generate(sample_strategy, words)

        decode_midi(generated, generated_path)
        decode_midi(words, original_path)


if __name__ == "__main__":
    main()
