import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse
import pandas as pd
import numpy as np
import glob
import os,re

nltk.download("stopwords")
nltk.download("wordnet")
cached = stopwords.words("english")
punctuation = [".", ",", "'", "\"", ":", ";", "?", "(", ")", "[", "]"]

def remove_stopwords(in_str):
    return " ".join([x for x in in_str.split() if x not in cached]) 

def remove_unwanted_chars(text: str):
    no_brackets = re.sub("([\(\[]).*?([\)\]])", "", text)
    no_digits = re.sub("\d+\.*\d*%*", "", no_brackets)
    no_newlines = re.sub("\n|\t", "", no_digits)
    no_punc = "".join([ch for ch in no_newlines if ch not in punctuation])
    return no_punc

def lemma(text: str,lemmatizer: WordNetLemmatizer):
    return " ".join([lemmatizer.lemmatize(x) for x in text.split()])

# Need to take in some CSV filepaths while specifying attributes and fnames
# and generate new filepaths with _clean appended and stopwords removed
if __name__ == "__main__":
    lemmatizer = WordNetLemmatizer()
    parser = argparse.ArgumentParser(description='Clean texts')
    parser.add_argument('folderpath',
                        help='Folder of CSV file to clean')
    parser.add_argument('output',
                        help='Output Folder')
    parser.add_argument('colname')
    args = parser.parse_args()
    colname = args.colname
    key = "publication_number"
    # Now read CSV file on every column name
    for filepath in glob.glob(os.path.join(args.folderpath,"*.csv")):
        df = pd.read_csv(filepath)
        new = []
        keys = []
        for _,x in df.iterrows():
                t = remove_unwanted_chars(x[colname])
                t = t.lower()
                t = remove_stopwords(t)     
                t = lemma(t,lemmatizer)
                new.append(t)
                keys.append(x[key])
        df_new = pd.DataFrame(data={key:keys,colname+"_clean":new})
        new_name = os.path.splitext(os.path.basename(filepath))[0] + "_clean.csv"
    # Save clean CSVs
        df_new.to_csv(os.path.join(args.output,new_name))
        