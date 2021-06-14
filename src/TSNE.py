import csv
import os

import numpy as np
import pdb
import pandas as pd
from sklearn.manifold import TSNE

# convert elements in colname to embeddings using key
def get_tsne(in_filename: str, model, out_filename: str, colname: str, key: str,proj_name: str) -> None:
    vec_rows = []
    df = pd.read_pickle(in_filename)
    # Take the column and run it through sbert. We assume that we are given a string with no '.'
    
    keys = df[key].to_list()
    projections = list(model.fit_transform(df[colname].to_list()))
    # for _,row in df.iterrows():
    #     if row.notna()[colname]:
    #         projections.append(model.encode(row[colname]))
    #         keys.append(row[key])
    df_new = pd.DataFrame(data={key:keys,proj_name:projections})
    df_new.set_index(key)
    df_new.to_pickle(out_filename)

def get_multi_tsne(in_filename: str, model, out_filename: str, colname: str, key: str,proj_name: str) -> None:
    vec_rows = []
    df = pd.read_pickle(in_filename)
    # Take the column and run it through sbert. We assume that we are given a string with no '.'
    
    keys = df[key].to_list()
    values_array = np.stack(df[colname].values)
    projections = list(model.fit_transform(values_array))
    df_new = pd.DataFrame(data={key:keys,proj_name:projections})
    df_new.set_index(key)
    df_new.to_pickle(out_filename)

def get_tsne_in_dir(dirname: str, out_dir: str, model, colname: str,key:str,proj_name: str) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            # for debugging
            print(filename)
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            get_tsne(full_path, model, new_path, colname,key,proj_name)


if __name__ == "__main__":
    n_components = 3
    tsne = TSNE(n_components=n_components)
    get_tsne_in_dir("/content/drive/MyDrive/NLP/Dataset/BERT/Abstract/Original",
    "/content/drive/MyDrive/NLP/Dataset/BERT/Abstract/tsne_3d",
    tsne,colname="BERT",key="publication_number",proj_name="tsne_3d")