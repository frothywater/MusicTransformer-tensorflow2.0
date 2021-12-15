import os
import pickle
import random

import numpy as np

import params
from processor import eos_word, pad_word


def shifted_sliding_pair(words: list, offset: int, length: int, pad_word: int, eos_word: int) -> tuple:
    if offset >= len(words):
        raise ValueError("Too short")
    x = np.array(words[offset : offset + length])
    y = np.array(words[offset + 1 : offset + 1 + length])
    if offset + 1 + length > len(words):
        # Tail of y exceeds
        y = np.concatenate([y, [eos_word]])
        if offset + length > len(words):
            # Tail of x exceeds
            current_length = len(x)
            x = np.concatenate([x, pad_word * np.ones(length - current_length)])
            y = np.concatenate([y, pad_word * np.ones(length - current_length)])
    return x, y


def get_offsets(length: int, max_length: int, density: int) -> list:
    offset_count = round(density * (length / max_length - 1) + 1)
    if offset_count <= 1:
        return [0]
    left_point_limit = length - max_length
    step = left_point_limit // (offset_count - 1)
    return list(range(0, left_point_limit + 1, step))


def pad_with_repetition(array: list, unit_length: int) -> list:
    result = array.copy()
    if len(array) % unit_length == 0:
        return array
    remain_count = (len(array) // unit_length + 1) * unit_length - len(array)
    extra = random.sample(result, remain_count)
    result += extra
    return result


def compile(words_path: str, dest_path: str):
    files = [file for file in os.listdir(words_path) if file.endswith(".pickle")]

    xs = []
    ys = []

    for filename in files:
        file_path = os.path.join(words_path, filename)
        with open(file_path, "rb") as file:
            words = pickle.load(file)

        word_length = len(words)
        for offset in get_offsets(word_length, params.max_seq, density=60):
            x, y = shifted_sliding_pair(words, offset, params.max_seq, pad_word, eos_word)
            xs.append(x)
            ys.append(y)

    pairs = list(zip(xs, ys))
    padded_pairs = pad_with_repetition(pairs, params.batch_size)
    random.shuffle(padded_pairs)
    x_list, y_list = zip(*padded_pairs)

    x_final = np.array(x_list).reshape(-1, params.batch_size, params.max_seq)
    y_final = np.array(y_list).reshape(-1, params.batch_size, params.max_seq)
    print(f"final: x={x_final.shape}, y={y_final.shape}")
    np.savez(dest_path, x=x_final, y=y_final)


def main():
    for dir_name in os.listdir(params.words_dir):
        print(f"Compiling {dir_name}...")
        words_path = os.path.join(params.words_dir, dir_name)
        dest_path = os.path.join(params.dataset_dir, dir_name + ".npz")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        compile(words_path, dest_path)


if __name__ == "__main__":
    main()
