- progress
- tensorflow-gpu
pip install progress tensorflow-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11

/mnt/nextlab/xinda/MusicGeneration/Paper_SKeleton_Framework/Dataset/v20220517/wikifornia/13_dataset_held50/Wikifornia_melody

```
python preprocess.py data/midi/train data/words/train && \
python preprocess.py data/midi/test data/words/test && \
python compile.py && \
python train.py

python generate.py
```