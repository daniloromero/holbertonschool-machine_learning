# 0x12. Transformer Applications

TF Datasets

For machine translation, we will be using the prepared Tensorflow Datasets ted_hrlr_translate/pt_to_en for English to Portuguese translation

To download Tensorflow Datasets, please use:
```
pip install --user tensorflow-datasets
```
To use this dataset:
```
$ cat load_dataset.py
#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
$ ./load_dataset.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity 