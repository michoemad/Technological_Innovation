#Motivation

Predicting technological innovation ahead of time is the key for making a breakthrough ina certain field.
For this reason, it has been highly sought after by inventors and companiesalike.
However, such a task is inherently difficult due to its innovative and chaotic nature.In  this  project,  we  aim  to  investigate  temporal  patterns  of 
technological  innovations  byusing patent data ranging from 1790 until today.  We first project patents into a higherdimensional space using Google’s Patent
BERT model and then utilize chaining methodsto predict the emergence of patents through time 

# Structure

preprocess.py: Cleans up patent data (abstract and claims) to be fed into the BERT model later on. This is done by:
Removing punctuation
Removing stop words
Converting words to lowercase
Lemmatizing
BERT_generate.py: Uses Google’s Patents BERT model to generate document embeddings by average word embeddings in either the abstract or claim section of a patent. Note that it is important to specify whether the abstract or claim is being used in the make_pubs_vectors() context argument since the model uses “[abstract]” and “[claim]” tokens to help generate more accurate embeddings [white paper].
TSNE.py: generates TSNE projections for the BERT embeddings obtained earlier. The dimension of the projections can be changed in the code by altering the n_components argument for TSNE. Note that TSNE can be really slow for large embeddings and hence we recommend using tsnecuda to speed things up. A tutorial can be found at https://colab.research.google.com/drive/1njTbpavLJl7K42OasJ2t492OfflDy9vm?usp=sharing
Fisher.py: the objective of this module is to calculate the Fisher Discriminant between every category in our mini dataset. More details can be found at X
