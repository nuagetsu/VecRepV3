'''
Utilities to use M-Trees with various metrics, max_node sizes, promotion and split policies
'''
import src.helpers.MetricUtilities as metrics

from mtree.mtree import MTree
import mtree.mtree as mtree

import pickle


def getKNearestNeighbours(tree, point, k):
    l = tree.search(point, k)
    imgs = list(l)
    return imgs

def getMTree(data, k, promote=mtree.M_LB_DIST_confirmed, partition=mtree.generalized_hyperplane, d=metrics.distance):
    tree = MTree(d, max_node_size=k, promote=promote, partition=partition)
    tree.add_all(data)
    return tree

def getMTreeFFT(data, k):
    tree = MTree(metrics.dist_fft, max_node_size=k)
    tree.add_all(data)
    return tree

def getMTreeFFTNumba(data, k, promote=mtree.M_LB_DIST_confirmed, partition=mtree.generalized_hyperplane):
    tree = MTree(metrics.dist_fft_numba, max_node_size=k, promote=promote, partition=partition)
    tree.add_all(data)
    return tree

def saveMTree(tree, filename):
    '''
    Saves tree to a pickle file
    '''
    with open(filename, "wb") as file:
        pickle.dump(tree, file)
    return 

def loadMTree(filename):
    '''
    Loads tree from a pickle file
    '''
    with open(filename, "rb") as file:
        tree = pickle.load(file)
    return tree