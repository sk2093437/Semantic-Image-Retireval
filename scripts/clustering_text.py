__author__ = 'TonySun'
"""
some clustering test on given context
"""

from sklearn.cluster import KMeans
import text_vectors
from sklearn.metrics import pairwise

# get filtered words and their vectors
word_and_vecs = text_vectors.getTextVectors()
vecs = []
words = []


# extract vectors
for key, value in word_and_vecs.items():
    vecs.append(value)
    words.append(key)

# Set "k" (num_clusters) to be 1/5th of the vocabulary size
num_clusters = len(vecs) / 5
#options: kmeans++, random or an ndarray
init_method = 'k-means++'

print("Begin k-means clustering...")
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
# obtain cluster IDs for each word
cluster_ids = kmeans_clustering.fit_predict(vecs)
print("Done...")
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(words, cluster_ids))

# For the first 10 clusters
for cluster in range(0, 10):
    # print the cluster number
    print("\nCluster %d" % cluster)
    # Find all of the words for that cluster number, and print them out
    r_words = []
    for i in range(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            r_words.append(word_centroid_map.keys()[i])

    print (r_words)



d = pairwise.pairwise_distances(vecs, metric='euclidean')
