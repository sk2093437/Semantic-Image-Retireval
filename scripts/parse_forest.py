__author__ = 'TonySun'

import utilites
from collections import Counter
import numpy as np
from scipy.io import loadmat
from build_supervised_tree import TreeNode
from build_supervised_tree import LabelCount
from operator import itemgetter


def load_forest():
    """
    Load forest file stored on the disk
    :return: loaded forest
    """
    return utilites.loadVariableFromFile("Corel5K/forest_400_trees_64_feats/forest.pkl")


def disp_tree_info(node):
    """
    Display all nodes contained in the given tree
    :param node: root node of a tree
    :return: nothing
    """
    print("")
    if node.parent is None:
        print("This is root node.")
    node.__str__()

    if not node.is_leaf:
        print("Upper child: ")
        disp_tree_info(node.upper_child)
        print("Lower child: ")
        disp_tree_info(node.lower_child)

    return ''


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


def parse_forest(sample, forest):
    """
    Feed test sample to all trees in the given forest then get
    statistical result of all trees
    :param sample: test sample
    :param forest: forest
    :return: summary of results of all trees
    """
    a_rc = []
    a_rs = []

    for tree in forest:
        rc, rs = parse_single_tree(sample, tree)
        a_rc.append(rc.label_count)
        a_rs = a_rs + list(rs)

    a_rc = np.asarray(a_rc)
    sum_a_rc = np.sum(a_rc, axis=0)  # get count in all trees for each concept
    sum_a_rs = Counter(a_rs)

    return sum_a_rc, sum_a_rs


# test code here
test_original = loadmat(utilites.getAbsPath('Corel5K/test_vectors_original.mat'))
test_original = test_original['test_vectors']
test_sample = test_original[0]

forest = load_forest()
# rc, rs = parse_single_tree(test_sample, forest[0])
# src: sum of concept count, srs: sum of retrieved sample count
src, srs = parse_forest(test_sample, forest)
label_name = range(100)
src_dict = dict(zip(label_name, src))

src_sorted = sorted(src_dict.items(), key=itemgetter(1))[::-1]
srs_sorted = sorted(srs.items(), key=itemgetter(1))[::-1]
src_top10 = src[0:10]
srs_top10 = srs_sorted[0:10]