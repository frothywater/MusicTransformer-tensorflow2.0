- progress
- tensorflow-gpu
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `

Do separately for training and valid dataset:
`python preprocess.py {midi_dir} {output_words_dir}`

```
python preprocess.py data/midi/train data/words/train
python preprocess.py data/midi/valid data/words/valid
python compile.py
python train.py
python generate.py
```