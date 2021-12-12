from  processor import pad_word, sos_word, eos_word, dictionary_size

max_seq = 512
embedding_dim = 256
num_layer = 6
batch_size = 60
dropout = 0.2
epochs = 100

pad_token = pad_word
token_sos = sos_word
token_eos = eos_word
vocab_size = dictionary_size

gpu_id = "2"
train_id = "wikifonia-melody-2"
model_dir = "data/model"
load_dir = "data/model"
generated_dir = "data/generated"
dataset_dir = "data/dataset"
