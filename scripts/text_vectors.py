__author__ = 'TonySun'
"""
This file is used to test the text vectors using Google's
# Word2Vec and Doc2Vec tools in Gensim toolkit
# till now the text file used is from annotation of Flickr 8k annotation dataset
# Each image have 5 annotations
# All the annotations are manually done and each one appears as a single sentence
"""
import gensim
import numpy as np
import numpy.matlib
import utilites
import setup
from sklearn.metrics import pairwise


# return text vectors calculated using Word2Vec by gensim
def getTextVectors():
    """
    open the annotation text file and read content
    build word vectors using Word2Vec and then extract
    the term/vector pairs into a dictionary
    :return: the ultimate word vectors
    """
    # get all filtered term(tag) names
    terms_corel5k_filtered = utilites.loadVariableFromFile("Corel5k/terms_corel5k_filtered.pkl")
    # get training image annotations: lists of separate terms
    train_anno_filtered = utilites.loadVariableFromFile("Corel5k/train_anno_filtered.pkl")
    # at first, we need to build a corpus for all tag-based annotations
    # we start by preparing a annotation matrix with all tags
    all_term_matrix = np.matlib.repmat(np.asarray(terms_corel5k_filtered), len(train_anno_filtered), 1)
    # then we keep tags for each image according to the values of their annotations: 1 keep, 0 drop
    corpus = []
    for an in range(len(train_anno_filtered)):
        corpus.append(list(all_term_matrix[an][np.where(train_anno_filtered[an] == 1)[0]]))


    # Set values for Word2Vec
    num_features = 300  # Use a 300-dimension vector to represent a word
    min_word_count = 5  # Word appears less than 5 times will be ignored
    num_workers = 4     # Number of threads to run in parallel
    context = 3        # Sample 5 words as input for each iteration

    # initialize a model using parameters above
    word_model = gensim.models.Word2Vec(workers=num_workers,
                               size=num_features, min_count=min_word_count, window=context)

    word_model.build_vocab(corpus) # build vocabulary on split sentenced
    print("Language model established.")
    print("Loading pre-trained language model...")
    # initialize the network weights using pre-trained model
    word_model.intersect_word2vec_format(utilites.getAbsPath(setup.lmodel_file_path), binary=True)
    print("Loaded weights from pre-trained Google News language model.")
    print("Training models...")
    # train the model to get word vectors
    word_model.train(corpus)
    print("Training completed.")

    """
    Calculate similarity matrix using given vectors
    We use pairwise distances to build the matrix
    :param model: a language model generated using Word2Vec
    :return: a similarity matrix
    """
    print ("Extracting word vectors...")
    print("Original number of words: " + str(len(word_model.index2word)))
    vecs = []
    terms_corel5k_filtered = utilites.loadVariableFromFile("Corel5k/terms_corel5k_filtered.pkl")
    # Index2word is a list that contains the names of the words in
    for word in terms_corel5k_filtered:
        vecs.append(word_model[word])  # now we extract all word vectors from the model

    print ("Term vectors haven been created.")
    d_pairwise_vecs = 1 - pairwise.pairwise_distances(vecs, metric='cosine')
    print("Similarity matrix has been built.")
    utilites.saveVariableToFile(d_pairwise_vecs, "Corel5k/tag_textual_similarity_matrix.pkl")

    return d_pairwise_vecs
