import tensorflow as tf
from processor import Event
import numpy as np

def bar_to_second(bar):
    return 60.0 / 120.0 * 4 * bar


def cropped_words(words: list, bar: int):
    result = []
    current_time = 0
    target_time = bar_to_second(bar)
    for word in words:
        event = Event.from_int(word)
        if event.type == "time_shift":
            current_time += event.value / 100
        if current_time > target_time:
            break
        result.append(word)
    return result

def softmax(logits, temp=1.0):
    logits -= logits.max()
    return tf.exp(logits / temp) / tf.sum(tf.exp(logits / temp))

def top_k(logits, k):
    result = tf.math.top_k(logits, k)
    indices = result.indices.numpy()
    probs = result.values.numpy()
    probs /= sum(probs)
    return np.random.choice(indices, size=1, p=probs)[0]
