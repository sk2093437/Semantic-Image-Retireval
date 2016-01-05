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
import utilites
import setup
# from sklearn.preprocessing import normalize


def filterWords(words_list):
    """
    mutual filtering the terms with the help of POS tags
    :param words_list: a list contains distinct words
    :return: filtered term list
    """
    words_list = set(words_list)
    # load variables
    words_and_depends = utilites.loadVariableFromFile()
    # it is a list containing 4 sets:
    # [0] obj set, [1] attribute set, [2] action set [3] dependency set
    # print len(words_and_depends[0])
    words_and_depends[0] = words_and_depends[0].intersection(words_list)
    words_and_depends[0] = utilites.removeStopWords(words_and_depends[0])

    words_and_depends[1] = words_and_depends[1].intersection(words_list)
    words_and_depends[1] = utilites.removeStopWords(words_and_depends[1])

    words_and_depends[2] = words_and_depends[2].intersection(words_list)
    words_and_depends[2] = utilites.removeStopWords(words_and_depends[2])


    # now we have all original words, and then we filter the terms in vectors
    # we compute intersection of these two sets to get their common words
    filtered_words = words_and_depends[0]
    # filtered_words = words_and_depends[0].union(words_and_depends[1]).union(words_and_depends[2])
    print("Number of words after filtering: " + str(len(filtered_words)))
    # now we filter word dependency and re-store them in file
    words_depens = words_and_depends[3].copy()

    for word_pair in words_depens:
        if (word_pair[0] not in filtered_words) | (word_pair[1] not in filtered_words):
            words_and_depends[3].remove(word_pair)  # only retain words in the filtered words list

    # save these thins in file again
    # utilites.saveVariableToFile(words_and_depends)

    return list(filtered_words)



def extractVecs(model):
    """
    :param model: a language model generated using Word2Vec
    :return: a dict contains pairs of term names and their associated feature vectors
    """
    print ("Extracting word vectors...")
    print("Original number of words: " + str(len(model.index2word)))
    feature_vecs = []

    filtered_words = filterWords(model.index2word)

    # Index2word is a list that contains the names of the words in
    for word in filtered_words:
        feature_vecs.append(model[word])  # now we extract all word vectors from the model

    # build a dictionary to use term name as index
    term_dict = dict(zip(filtered_words, np.array(feature_vecs)))
    print ("Single term vectors haven been created.")


    """
    depend_vecs = []
    # now we add word dependencies as terms and extract their vectors
    wd = list(utilites.loadVariableFromFile()[3])  # load dependencies
    for pair in wd:
        # get two vectors in each dependency pair
        v_temp1 = model[pair[0]]
        v_temp2 = model[pair[1]]
        # average the vectors and L2-normalize the result as dependency vectors
        depend_vecs.append((v_temp1 + v_temp2)/2)

    word_phrases = utilites.tupleToPhrase(wd)
    phrase_dict = dict(zip(word_phrases, normalize(np.array(depend_vecs))))
    term_dict.update(phrase_dict)
    """

    return term_dict


# return text vectors calculated using Word2Vec by gensim
def getTextVectors():
    """
    open the annotation text file and read content
    build word vectors using Word2Vec and then extract
    the term/vector pairs into a dictionary
    :return: the ultimate word vectors
    """
    raw_text = utilites.get_raw_text()

    # now we only need the annotations
    annotations = [line[2] for line in raw_text]

    # Prepare the sentences
    sentences = utilites.annotation_to_wordlists(annotations)

    # Set values for Word2Vec
    num_features = 300  # Use a 300-dimension vector to represent a word
    min_word_count = 5  # Word appears less than 5 times will be ignored
    num_workers = 4     # Number of threads to run in parallel
    context = 5        # Sample 5 words as input for each iteration

    # initialize a model using parameters above
    word_model = gensim.models.Word2Vec(workers=num_workers,
                               size=num_features, min_count=min_word_count, window=context)

    word_model.build_vocab(sentences) # build vocabulary on split sentenced
    print("Language model established.")
    print("Loading pre-trained language model...")
    # initialize the network weights using pre-trained model
    word_model.intersect_word2vec_format(utilites.getAbsPath(setup.lmodel_file_path), binary=True)
    print("Loaded weights from pre-trained Google News language model.")
    print("Training models...")
    # train the model to get word vectors
    word_model.train(sentences)
    print("Training completed.")

    return extractVecs(word_model)




	

