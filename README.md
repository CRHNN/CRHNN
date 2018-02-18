# Collaborative Ranking with Hierarchical Neural Network for Hybrid Recommender Systems
### Introduction
Traditional collaborative filtering (CF) methods suffer from limitations in many cases, as they generate recommendations by purely mining the user-item rating matrix and even the ratings are often very sparse in many applications. To provide more personalized and accurate recommendations, hybrid CF methods were proposed to combine standard CF and content-based techniques, which can simultaneously utilize the rating information and auxiliary information such as textual information. However, most of them treat a document as a word-count matrix or one sequence of words while ignoring the obvious hierarchical structure of the textual information. 


To address this problem, we present a novel Collaborative Ranking with Hierarchical Neuron Network (CRHNN), which comprises of word2vec, sent2vec and doc2vec layers from bottom to top to capture the hierarchical structure of textual information in the CF setting. CRHNN utilizes the dynamic routing protocol to encapsulate sentence vectors into one document vector, and the dynamic routing protocol can be viewed as a novel attention mechanism which can be trained efficiently via stochastic gradient descent. To simultaneously modeling the hierarchical structure of the document for the content textual information and the user feedback information in CF setting, we design a novel loss that consists of two terms. One is the reconstruction loss that measures' the expressiveness of sentence vectors, and the other is the ranking loss that measures the fitness of the model for user actions. Empirical experiments on three real-world datasets from different domains (CiteULike and Movielens) indicate that CRHNN can provide more effective recommendation than state-of-the-art hybrid CF methods.
 
### Dataset
The CiteULike dataset we used in our paper is from [Collaborative topic modeling for recom-mending scientific articles]. MovPlot1M and MovPlot10M are from [Collaborative topic regression for online recommender systems: an online and Bayesian approach]. 

### How to run
1.download dataset.

2.python train/train_deepMF_with_text.py 
