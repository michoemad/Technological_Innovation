import numpy as np
import pandas as pd
import os
from itertools import combinations

def euclid(x,y) -> float:
    return np.sum(np.square(x-y))

# X and Y and lists of points
def compute_fisher_two_classes(X,Y):
    mu_X = np.mean(X,axis=0)
    mu_Y = np.mean(Y,axis=0)
    mu = (np.sum(X,axis=0) + np.sum(Y,axis=0)) / (X.shape[0]+Y.shape[0])
    Vb = euclid(mu_X,mu) + euclid(mu_Y,mu)
    Vw = 0
    for x in X:
        Vw += euclid(x,mu_X)
    for y in Y:
        Vw += euclid(y,mu_Y)
    return (Vb/Vw)

def create_matrix(directory,colname):
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and "rejected" not in filename:
            names.append(os.path.splitext(filename)[0])
    frame = pd.DataFrame(columns=names,index=names)
    for comb in combinations(names,2):
        # print(comb)
        # compute Fisher
        C1 = pd.read_pickle(os.path.join(directory,comb[0]+".csv"))
        C2 = pd.read_pickle(os.path.join(directory,comb[1]+".csv"))
        num = compute_fisher_two_classes(C1[colname],C2[colname])
        frame[comb[0]][comb[1]]= num
        frame[comb[1]][comb[0]]= num
    return frame

def create_random_matrix(directory,colname,n):
    names = []
    values = dict()
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and "rejected" not in filename:
            names.append(os.path.splitext(filename)[0])
            values[names[-1]] = pd.read_pickle(os.path.join(directory,filename)).sample(n=n)
    frame = pd.DataFrame(columns=names,index=names)
    # print(values[names[-1]].keys())
    for comb in combinations(values,2):
        # compute Fisher
        num = compute_fisher_two_classes(values[comb[0]][colname],values[comb[1]][colname])
        frame[comb[0]][comb[1]]= num
        frame[comb[1]][comb[0]]= num
    return frame

if __name__ == "__main__":
    # An example
    DIR = "/content/drive/MyDrive/NLP/Dataset/sbert/Claim/Original"
    OUTPUT_DIR=  "/content/drive/MyDrive/NLP/Dataset/sbert/Claim/Random_Fisher"
    colname = "claims_clean"
    # for i in range(100):
    #     matrix = create_random_matrix(DIR,colname,200)
    #     new_path = os.path.join(OUTPUT_DIR,"sbert_claim_rand_"+str(i)+".csv")
    #     matrix.to_pickle(new_path)
   # X = [[1,0],[2,3]]
    #Y = [[-1,0],[-2,3]]
    # print(pd.read_pickle(os.path.join(OUTPUT_DIR,"sbert_claim_rand_79.csv")))
    #print(compute_fisher_two_classes(X,Y))