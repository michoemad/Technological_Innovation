import nltk
from nltk.corpus import stopwords
import argparse
import pandas as pd
import numpy as np
import glob
import os

nltk.download("stopwords")
cached = stopwords.words("english")
def remove_stopwords(in_str):
    return " ".join([x for x in in_str.split() if x not in cached]) 

# Need to take in some CSV filepaths while specifying attributes and fnames
# and generate new filepaths with _clean appended and stopwords removed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean texts')
    parser.add_argument('folderpath',
                        help='Folder of CSV file to clean')
    parser.add_argument('output',
                        help='Output Folder')
    parser.add_argument('colname')
    args = parser.parse_args()
    colname = args.colname
    # Now read CSV file on every column name
    for filepath in glob.glob(args.folderpath+"*.csv"):
        df = pd.read_csv(filepath)
        new = []
        # Hack, can remove
        if (colname+"_clean" in df.keys()):
            continue
        for _,x in df.iterrows():
            if x.notna()[colname]:
                new.append(remove_stopwords(x[colname]))
            else:
                new.append(np.nan)
        df[colname] = new
        new_name = os.path.splitext(os.path.basename(filepath))[0] + ".csv"
        df = df.rename(columns={colname:colname+"_clean"})
    # Save clean CSVs
        df.to_csv(args.output+new_name)
        