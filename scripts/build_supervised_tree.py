from __future__ import division
import math
import numpy as np
import utilites
from scipy.io import loadmat
import copy

__author__ = 'TonySun'


import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


class Data(object):
    """
    Data class to represent splitting samples
    """
    def __init__(self):
        self.samples = None  # feature vectors
        self.labels = None   # annotations
        self.features = None # original feature index
        self.orig_sample_indexes = None   # original sample index before selection

class TreeNode(object):
    """
    A node of the tree inside the random forest
    """
    def __init__(self, parent):
        self.is_leaf = False           # if it is a leaf node
        self.feat_split_local = None
        self.feat_split_index = None
        self.feat_split_value = None
        self.num_samples = -1
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.height = None            # the depth of this node (which layer in the tree)
        self.orig_sample_indexes = None
        self.label_count = None


    def __str__(self):
        """
        return string representation of this node
        :return:
        """
        print("Node depth: " + str(self.height))
        print("Number of samples locate here: ") + str(self.num_samples)
        print("Split dimension index: " + str(self.feat_split_index))
        print("Threshold is: " + str(self.feat_split_value))
        if self.is_leaf == True:
            print("This is a leaf Node.")
            print("Original index of samples: " + str(self.orig_sample_indexes))

        return ""


class LabelCount:
    """
    Count labels for given image samples
    """
    def __init__(self, label_names, label_count):
        self.label_names = label_names
        self.label_count = label_count


def count_labels(labels):
    """
    use collections.Count to count the histogram of labels
    :param labels: labels of image samples
    :return: label names and corresponding counts
    """
    # and return label names and label count
    label_names = range(len(labels[0]))  # use label index as names
    label_count = np.sum(labels, axis=0)   # sum by column to get counts for each label
    return LabelCount(label_names, label_count)


def cal_data_entropy(dataset, label_counts):
    """
    Calculate the entropy of the current dataset
    :param dataset: dataset
    :param label_counts: label counts related to given dataset
    :return:
    """
    num_samples = float(len(dataset.samples))
    entropy = 0

    # for each label
    for i in range(0, len(label_counts.label_names)):
        # calculate its ratio in the dataset
        prob = label_counts.label_count[i] / num_samples
        if prob != 0:
            # calculate standard entropy
            entropy += prob * math.log(prob, 2)
            # entropy += prob * math.log(prob, base=len(label_counts.label_names))

    entropy = -entropy
    return entropy


def cal_info_gain(dataset, entropy, thres, fi):
    """
    Calculate the gain of a particular feature split
    :param dataset: dataset
    :param entropy: entropy of the dataset
    :param thres: threshold of the split location
    :param feat_dims: location of the split
    :return: information gain of this split (double)
    """
    feat_entropy = 0.0
    num_samples = int(len(dataset.samples))
    gain_upper_dataset = Data()
    gain_lower_dataset = Data()
    gain_upper_dataset.features = dataset.features  # obtain dataset before split
    gain_lower_dataset.features = dataset.features

    # column is the feature
    upper_sample_index = np.array(np.where(dataset.samples[:, fi] >= thres)[0])
    gain_upper_dataset.samples = dataset.samples[upper_sample_index]
    gain_upper_dataset.labels = dataset.labels[upper_sample_index]

    lower_sample_index  = np.array(np.where(dataset.samples[:, fi] < thres)[0])
    gain_lower_dataset.samples = dataset.samples[lower_sample_index]
    gain_lower_dataset.labels = dataset.labels[lower_sample_index]

    if len(gain_upper_dataset.samples) == 0 or len(gain_lower_dataset.samples) == 0:
        # If it tries to split on the max or min of the feature value's range
        return -1

    upper_hist = count_labels(gain_upper_dataset.labels)  # count labels occurrence of dataset
    lower_hist = count_labels(gain_lower_dataset.labels)
    feat_entropy += cal_data_entropy(gain_upper_dataset, upper_hist) * len(gain_upper_dataset.samples) / num_samples
    feat_entropy += cal_data_entropy(gain_lower_dataset, lower_hist) * len(gain_lower_dataset.samples) / num_samples

    return entropy - feat_entropy   # return information gain


"""
After we defined every component, we can recursively generate a single tree in random forest
Initialize Tree
Stopping criteria: reach max depth, or dataset it pure

for all features f in dataset
    compute information gain if we split on it
f_best = best feature split location according to the criteria
tree = create a decision node that test f_best in the root
dv (v = 1,2,3) = induced sub datasets from Data based on f_best

for all dv
    tree = generate_supervised_tree(dv)
    attach tree to the corresponding branch of tree
return tree
"""

def generate_supervised_tree(dataset, parent_node, max_depth, min_leaf_sample):
    """
    Generate a supervised tree recursively
    :param dataset: dataset used
    :param parent_node: parent node of current tree node
    :param max_depth: max depth of the generated tree, default=12
    :param min_leaf_sample: number of min samples in leaf node, default=4
    :return: single generated tree
    """
    # select square root of number of features
    n_feats = int(round(math.sqrt(len(dataset.features))))
    # partly copy feature index to selected dataset
    ran_feat_index = np.random.randint(0, len(dataset.features), size=n_feats)

    # make sure the selected features do not contain duplicates
    while True:
        if len(set(ran_feat_index)) < n_feats:
            ran_feat_index = np.random.randint(0, len(dataset.features), size=n_feats)
        else:
            break

    # copy selected feature index
    selected_feat_index = np.copy(dataset.features[ran_feat_index])

    node = TreeNode(parent_node)  # generate a new Node
    if parent_node is None:
        node.height = 0  # make it root node if no parent node exists
    else:
        node.height = node.parent.height + 1 # add depth

    node.num_samples = len(dataset.samples)  # get number of samples ind dataset
    label_counts = count_labels(dataset.labels)    # get label counts for given dataset

    # if all samples contain the same label make it a leaf node
    for i in range(len(label_counts.label_names)):
        if len(dataset.samples) == label_counts.label_count[i]:
            node.is_leaf = True
            node.num_samples = len(dataset.samples)
            node.label_count = count_labels(dataset.labels)  # save label info (important)
            node.orig_sample_indexes = dataset.orig_sample_indexes
            print(node.__str__())
            return node
        else:
            node.is_leaf = False


    feat_to_split = None  # Global index of feature which will be used to split data
    fi_to_split = None    # local index of feature we will split on
    glob_max_gain = 0.0  # the gain given by the best attribute
    glob_split_thres = None   # the threshold to split data
    glob_min_gain = 0.001  # global info gain
    dataset_entropy = cal_data_entropy(dataset, label_counts)  # get entropy of dataset before split


    # begin to select optimal feature to split on
    for fi in range(len(selected_feat_index)):
        feat_index = selected_feat_index[fi]  # get the original feature index in all features

        local_max_gain = 0.0
        local_split_thres = 0.0
        # we find all values of this feature, which should be one column of the dataset
        feat_value_vector = dataset.samples[:, feat_index]

        f_min = min(feat_value_vector)
        f_max = max(feat_value_vector)
        threshold_vector = f_min+(f_max-f_min) * np.random.random(size=15)

        for val in list(threshold_vector):
            # calculate the info gain if we split on this value
            # if info gain is larger then local_max_gain, save the split location and the value
            local_gain = cal_info_gain(dataset, dataset_entropy, val, feat_index)

            if local_gain > local_max_gain:
                local_max_gain = local_gain  # save gain and threshold if info gain increases
                local_split_thres = val

        print("Max gain for feature " + str(feat_index) + " is " + str(local_max_gain))

        if local_max_gain > glob_max_gain:
            glob_max_gain = local_max_gain  # save global max info gain
            glob_split_thres = local_split_thres  # save global splitting threshold
            feat_to_split = feat_index  # save the original feature index to split on
            fi_to_split = fi  # save local feature index

        # after the loop, we know which feature is the best to split on
        if glob_split_thres is None or feat_value_vector is None:
            node.is_leaf = True
            node.num_samples = len(dataset.samples)
            node.label_count = count_labels(dataset.labels)
            node.orig_sample_indexes = dataset.orig_sample_indexes
            print(node.__str__())
            print("Couldn't find a feature to split on or a split threshold")
            return node

        elif glob_max_gain <= glob_min_gain or node.height >= max_depth or node.num_samples <= min_leaf_sample:
            node.is_leaf = True
            node.num_samples = len(dataset.samples)
            node.label_count = count_labels(dataset.labels)
            node.orig_sample_indexes = dataset.orig_sample_indexes
            print(node.__str__())
            return node

        # end of inner loop
    # end of outer loop

    # if no leaf node is created, we give values to node's attribute the continue searching
    node.feat_split_index = feat_to_split
    node.feat_split_local = fi_to_split
    node.feat_split_value = glob_split_thres
    print(node.__str__())

    upper_dataset = Data()  # initialize datasets for children
    lower_dataset = Data()
    upper_dataset.features = dataset.features  # prepare all features for selection on next spit
    lower_dataset.features = dataset.features  # not that for each split, m features are randomly selected

    if feat_to_split is not None:
        upper_index = np.where(dataset.samples[:, fi_to_split] >= glob_split_thres)
        upper_dataset.samples = dataset.samples[upper_index]
        upper_dataset.labels = dataset.labels[upper_index]
        upper_dataset.orig_sample_indexes = dataset.orig_sample_indexes[upper_index]

        lower_index = np.where(dataset.samples[:, fi_to_split] < glob_split_thres)
        lower_dataset.samples = dataset.samples[lower_index]
        lower_dataset.labels = dataset.labels[lower_index]
        lower_dataset.orig_sample_indexes = dataset.orig_sample_indexes[lower_index]


    # now we split dataset into upper and lower halves, we continue to generate tree node
    node.upper_child = generate_supervised_tree(upper_dataset, node, max_depth, min_leaf_sample)
    node.lower_child = generate_supervised_tree(lower_dataset, node, max_depth, min_leaf_sample)

    return node  # return the root node when all nodes generated


def parse_single_tree(sample, node):
    """
    parse an new instance using given tree
    :param sample: test instance
    :param node: single tree, root node
    :return: training sample names related to this instance and their label count
    """
    if node.is_leaf:
        # return sample names and corresponding label count of this node
        return (node.label_count, node.orig_sample_indexes)
    else:
        if sample[node.feat_split_index] >= node.feat_split_value:
            return parse_single_tree(sample, node.upper_child)
        else:
            return parse_single_tree(sample, node.lower_child)


def disp_tree(node):
    print("")
    if node.parent is None:
        print("This is root node.")

    node.__str__()
    if node.is_leaf == False:
        disp_tree(node.upper_child)
        disp_tree(node.lower_child)
    return ""



def generate_random_forest(dataset, n_trees,  perc_samples=0.66, max_depth=12, max_leaf_sample=4):
    """
    Build random forest
    1. Sample N random samples with replacement to create a subset of the data.
       The subset it about 66% of the whole dataset
    2. At each node:
       (1). For some number m, m feature variables are selected at random from all feature variables
       (2). The feature variable that provides the best split, according to some objective function
            (here we use max info gain), is used to do a binary split on the node.
       (3). At the next node, choose another m variables at random from all feature variables and
            do the same.
       The choice of m is generally 1/2 * sqrt(m), sqrt(m) and 2 * sqrt(m)
    """
    forest = []
    # we select 66% of data samples by default
    n_samples = int(round(len(dataset.samples) * perc_samples))
    selected_dataset = Data()

    # start to build forest, for each tree, we select samples and features
    # then build a tree using given dataset
    for i in range(n_trees):
        # generate random integers from 0 to last sample as random indexes
        ran_sam_index = np.random.randint(0, len(dataset.samples), size=n_samples)

        # partly copy data to selected dataset
        selected_dataset.samples = np.copy(dataset.samples[ran_sam_index])
        selected_dataset.labels = np.copy(dataset.labels[ran_sam_index])
        selected_dataset.orig_sample_indexes = np.array(ran_sam_index)
        selected_dataset.features = np.copy(dataset.features)

        tree_tmp = generate_supervised_tree(selected_dataset, None, max_depth, max_leaf_sample)
        forest.append(tree_tmp)

        utilites.saveVariableToFile(tree_tmp, "Corel5K/forest/tree_" + str(i) + ".pkl")

    utilites.saveVariableToFile(forest, "Corel5K/forest.pkl")

    return forest



"""
Test code
"""
train_original = loadmat(utilites.getAbsPath('Corel5K/train_vectors_original.mat'))
train_original = train_original['train_vectors']
train_label = utilites.loadVariableFromFile(utilites.getAbsPath("Corel5k/train_anno_concept.pkl"))

# prepare data
train_data = Data()
train_data.samples = train_original
train_data.labels = train_label
train_data.orig_sample_indexes = np.array(range(len(train_original)))
train_data.features = np.array(range(np.shape(train_original)[1]))

tic()
rand_forest = generate_random_forest(train_data, 1)
toc()












