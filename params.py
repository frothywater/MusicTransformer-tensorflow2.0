from processor import dictionary_size, eos_word, pad_word, sos_word

max_seq = 2048
embedding_dim = 256
num_layer = 6
batch_size = 5
dropout = 0.2
epochs = 200

pad_token = pad_word
token_sos = sos_word
token_eos = eos_word
vocab_size = dictionary_size

gpu_id = "2"
train_id = "wikifonia-0316"
model_dir = "data/model"
load_dir = "data/model"
load_epoch = 47
words_dir = "data/words"
dataset_dir = "data/dataset"
generated_dir = "data/generated"
