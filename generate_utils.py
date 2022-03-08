import os

import numpy as np

from processor import unit_per_bar, word2event


def get_test_files(words_dir: str) -> list:
    test_dir = os.path.join(words_dir, "test")
    return [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith(".pickle")]


def bar_length(words: list):
    result = 0.0
    for word in words:
        event = word2event[word]
        if event.type == "time_shift":
            result += event.value / unit_per_bar
    return result


def cropped_words(words: list, bar: int):
    result = []
    current_time = 0
    target_time = bar * unit_per_bar
    for word in words:
        event = word2event[word]
        if event.type == "time_shift":
            current_time += event.value
        if current_time > target_time:
            break
        result.append(word)
    return result


def softmax(logits, temp=1.0):
    logits -= logits.max()
    return np.exp(logits / temp) / np.sum(np.exp(logits / temp))


def top_k(probs, k):
    sorted_index = np.argsort(probs)[::-1]
    candi_index = sorted_index[:k]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    return np.random.choice(candi_index, size=1, p=candi_probs)[0]


class SampleStrategy:
    def __init__(self, temp: float, k: int):
        self.temp = temp
        self.k = k
