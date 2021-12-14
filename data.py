import os
import pickle
import random

import numpy as np

import params as par
import utils


class Data:
    def __init__(self, dir_path):
        path_train = os.path.join(dir_path, "train")
        path_valid = os.path.join(dir_path, "valid")
        path_test = os.path.join(dir_path, "test")
        self.file_dict = {
            "train": list(utils.find_files_by_extensions(path_train, [".pickle"])),
            "eval": list(utils.find_files_by_extensions(path_valid, [".pickle"])),
            "test": list(utils.find_files_by_extensions(path_test, [".pickle"])),
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        self.file_count = len(self.file_dict["train"]) + len(self.file_dict["eval"]) + len(self.file_dict["test"])

    def __repr__(self):
        return f"file_count={self.file_count}"

    def batch(self, batch_size, length, mode="train"):
        batch_files = random.sample(self.file_dict[mode], k=batch_size)
        batch_data = [self._get_seq(file, length) for file in batch_files]
        return np.array(batch_data)  # batch_size, seq_len

    def seq2seq_batch(self, batch_size, length, mode="train"):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length]
        y = data[:, length:]
        return x, y

    def smallest_encoder_batch(self, batch_size, length, mode="train"):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, : length // 100]
        y = data[:, length // 100 : length // 100 + length]
        return x, y

    def slide_seq2seq_batch(self, batch_size, length, mode="train"):
        data = self.batch(batch_size, length + 1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def random_sequential_batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = []
        for i in range(batch_size):
            data = self._get_seq(batch_files[i])
            for j in range(len(data) - length):
                batch_data.append(data[j : j + length])
                if len(batch_data) == batch_size:
                    return batch_data

    def sequential_batch(self, batch_size, length):
        batch_data = []
        data = self._get_seq(self.files[self._seq_file_name_idx])

        while len(batch_data) < batch_size:
            while self._seq_idx < len(data) - length:
                batch_data.append(data[self._seq_idx : self._seq_idx + length])
                self._seq_idx += 1
                if len(batch_data) == batch_size:
                    return batch_data

            self._seq_idx = 0
            self._seq_file_name_idx = self._seq_file_name_idx + 1
            if self._seq_file_name_idx == len(self.files):
                self._seq_file_name_idx = 0
                print("iter intialized")

    def _get_seq(self, fname, max_length=None):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        if max_length is not None:
            if len(data) > max_length:
                start = random.randrange(0, len(data) - max_length)
                data = data[start : start + max_length]
            elif len(data) < max_length:
                data = np.append(data, par.token_eos)
                while len(data) < max_length:
                    data = np.append(data, par.pad_token)
        return data
