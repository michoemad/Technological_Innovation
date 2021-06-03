import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
import pdb
import pandas as pd

# convert elements in colname to embeddings using key
def make_pubs_vectors(in_filename: str, model: SentenceTransformer, out_filename: str, colname: str, key: str) -> None:
    vec_rows = []
    df = pd.read_csv(in_filename)
    # Take the column and run it through sbert. We assume that we are given a string with no '.'
    
    projections = []
    keys = []
    for _,row in df.iterrows():
        if row.notna()[colname]:
            projections.append(model.encode(row[colname]))
            keys.append(row[key])
    proj_name = "projections_{}".format(colname)
    df_new = pd.DataFrame(data={key:keys,colname:projections})
    df_new.set_index(key)
    df_new.to_pickle(out_filename)

def make_pubs_vectors_in_dir(dirname: str, out_dir: str, model: SentenceTransformer, colname: str,key:str) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            # for debugging
            print(filename)
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            make_pubs_vectors(full_path, model, new_path, colname,key)


if __name__ == "__main__":
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    make_pubs_vectors_in_dir("/content/drive/MyDrive/NLP/Dataset/clean","/content/drive/MyDrive/NLP/Dataset/sbert/Claim/Original",
    model,"claims_clean","publication_number")
    # make_pubs_vectors("/content/drive/MyDrive/NLP/Dataset/clean/C07H5_clean.csv",model,"temp.csv","text_clean","publication_number")
