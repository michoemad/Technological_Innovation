import nltk
from nltk.corpus import stopwords
import argparse
import pandas as pd
import glob
cached = stopwords.words("english")
def remove_stopwords(in_str):
    return " ".join([x for x in in_str.split() if x not in cached]) 

# Need to take in some CSV files while specifying attributes and fnames
# and generate new files with _clean appended and stopwords removed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean texts')
    parser.add_argument('folderpath',
                        help='Folder of CSV files to clean')
    parser.add_argument('output',
                        help='Output Folder')
    parser.add_argument('colnames', nargs="+")
    args = parser.parse_args()
    # Now read CSV files
    for file in glob.glob(args.folderpath+"*.csv"):
        df = pd.read_csv(file)
        new = []
        for _,x in df.iterrows():
            new.append(remove_stopwords(x[args.colnames[0]]))
    # Save clean CSVs
    