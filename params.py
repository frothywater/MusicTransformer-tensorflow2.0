from processor import dictionary_size, eos_word, pad_word, sos_word

max_seq = 512
embedding_dim = 256
num_layer = 6
batch_size = 50
dropout = 0.2
epochs = 500

pad_token = pad_word
token_sos = sos_word
token_eos = eos_word
vocab_size = dictionary_size

gpu_id = "2"
train_id = "wikifonia-melody-v2"
model_dir = "data/model"
load_dir = None
generated_dir = "data/generated"
dataset_dir = "data/dataset"
