import os
import collections
import math
import random
import sys
import time
from typing import Dict, List, Tuple
from sklearn.metrics import pairwise
from sklearn.manifold import TSNE
# Use Tensorflow 2.0
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load the BERT model
BERT_REPO_PATH = "/content/drive/MyDrive/NLP/my_bert"
if not BERT_REPO_PATH in sys.path:
  sys.path += [BERT_REPO_PATH]
import tokenization

# Load model
MAX_SEQ_LENGTH = 512
MODEL_DIR = "/content/drive/MyDrive/NLP/BERT/"
VOCAB = '/content/drive/MyDrive/NLP/BERT/assets.extra/vocab.txt'
model = tf.compat.v2.saved_model.load(export_dir=MODEL_DIR, tags=['serve'])
model = model.signatures['serving_default']

tokenizer = tokenization.FullTokenizer(VOCAB, do_lower_case=True)
# Mean pooling layer for combining
pooling = tf.keras.layers.GlobalAveragePooling1D()

def get_bert_token_input(texts,context):
  input_ids = []
  input_mask = []
  segment_ids = []

  for text in texts:
    tokens = tokenizer.tokenize(text)
    if len(tokens) > MAX_SEQ_LENGTH - 2:
      tokens = tokens[0:(MAX_SEQ_LENGTH - 2)]
    # add to front and back
    tokens = ['[CLS]'] + [context] + tokens + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    token_pad = MAX_SEQ_LENGTH - len(ids)
    # masking
    input_mask.append([1] * len(ids) + [0] * token_pad)
    input_ids.append(ids + [0] * token_pad)
    segment_ids.append([0] * MAX_SEQ_LENGTH)
  
  return {
      'segment_ids': tf.convert_to_tensor(segment_ids, dtype=tf.int64),
      'input_mask': tf.convert_to_tensor(input_mask, dtype=tf.int64),
      'input_ids': tf.convert_to_tensor(input_ids, dtype=tf.int64),
      'mlm_positions': tf.convert_to_tensor([], dtype=tf.int64)
  }

def get_BERT_embedding(text,context):
  inputs = get_bert_token_input([text],context)
  response = model(**inputs)
  avg_embeddings = pooling(
      tf.reshape(response['encoder_layer'], shape=[1, -1, 1024]))
  return avg_embeddings.numpy()[0]

# convert elements in colname to embeddings using key
def make_pubs_vectors(in_filename: str, model, out_filename: str, colname: str, key: str,context: str) -> None:
    vec_rows = []
    df = pd.read_csv(in_filename)
    # Take the column and run it through sbert. We assume that we are given a string with no '.'
    projections = []
    keys = []
    for _,row in df.iterrows():
        if row.notna()[colname]:
            projections.append(get_BERT_embedding(row[colname],context))
            keys.append(row[key])
    proj_name = "projections_{}".format(colname)
    df_new = pd.DataFrame(data={key:keys,colname:projections})
    df_new.set_index(key)
    df_new.to_pickle(out_filename)

def make_pubs_vectors_in_dir(dirname: str, out_dir: str, model, colname: str,key:str,context:str) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            # for debugging
            print(filename)
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            make_pubs_vectors(full_path, model, new_path, colname,key,context)

if __name__ == "__main__":
  make_pubs_vectors_in_dir("/content/drive/MyDrive/NLP/Dataset/clean","/content/drive/MyDrive/NLP/Dataset/BERT/Abstract/Original",
model,"text_clean","publication_number","[abstract]")