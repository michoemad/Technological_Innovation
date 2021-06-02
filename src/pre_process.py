import nltk
from nltk.corpus import stopwords
import argparse
import pandas as pd
import glob
import os
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
        for _,x in df.iterrows():
            new.append(remove_stopwords(x[colname]))
        df[colname] = new
        new_name = os.path.splitext(os.path.basename(filepath))[0] + "_clean.csv"
        df = df.rename(columns={colname:colname+"_clean"})
    # Save clean CSVs
        df.to_csv(args.output+new_name)
        