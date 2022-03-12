- progress
- tensorflow-gpu
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `

Do separately for training and valid dataset:
`python preprocess.py {midi_dir} {output_words_dir}`

```
python preprocess.py data/13_data_otherBaselines_split82/skeleton/train data/words/train
python preprocess.py data/13_data_otherBaselines_split82/skeleton/valid data/words/valid
python preprocess.py data/13_data_otherBaselines_split82/skeleton/test data/words/test
python compile.py
python train.py
python generate.py
```