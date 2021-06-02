def get_bert_token_input(texts):
  input_ids = []
  input_mask = []
  segment_ids = []

  for text in texts:
    tokens = tokenizer.tokenize(text)
    if len(tokens) > MAX_SEQ_LENGTH - 2:
      tokens = tokens[0:(MAX_SEQ_LENGTH - 2)]
    # add to front and back
    tokens = ['[CLS]'] + tokens + ['[SEP]']
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

def get_BERT_embedding(text):
  inputs = get_bert_token_input([text])
  response = model(**inputs)
  avg_embeddings = pooling(
      tf.reshape(response['encoder_layer'], shape=[1, -1, 1024]))
  return avg_embeddings.numpy()[0]

# Give the column name of the text to project
def get_df_embeddings(df,colname):
  docs_embeddings = []
  for _, row in df.iterrows():
    docs_embeddings.append(get_BERT_embedding(row[colname]))
  return docs_embeddings