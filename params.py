import processor as sequence

max_seq = 512
embedding_dim = 256
num_layer = 6
batch_size = 60
dropout = 0.2
epochs = 100

event_dim = sequence.RANGE_NOTE_ON + sequence.RANGE_NOTE_OFF + sequence.RANGE_TIME_SHIFT + sequence.RANGE_VEL
pad_token = event_dim
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3

gpu_id = "2"
train_id = "wikifonia-melody-2"
model_dir = "data/model"
load_dir = "data/model"
generated_dir = "data/generated"
dataset_dir = "data/dataset"
