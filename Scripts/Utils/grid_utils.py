from sys import stderr
from gensim.models import FastText  # only in get_fasttext_vectors()
from collections import Counter  # only in custom_cluster() and mask_low_freq()
from sklearn.cluster import KMeans 
import numpy as np


class Averager(object):
    def __init__(self):
        self.numerator = 0.0
        self.denominator = 0.0
        self.average = 0.0
    def increment_numerator(self, increment):
        self.numerator += increment
    def increment_denominator(self, increment):
        self.denominator += increment
    def get_average(self):
        if self.denominator:
            self.average = self.numerator / self.denominator


def get_fasttext_vectors(wfs, sents, model='skipgram', size=200, min_count=1, epochs=5, inductive_bias='semantic', masked_embeddings=False, target_affix_embeddings=False, target_syntactic_windows=False):
    # Adjust parameters according to the desired inductive bias
    if inductive_bias == 'syntactic':
        window, min_n, max_n, n_freq_vocab = 5, 3, 6, 250
        if target_syntactic_windows:
            window = 1
            stderr.write('\ttargeting small context windows to capture syntactic phenomena..\n')
        if target_affix_embeddings:
            min_n, max_n = 2, 4
            stderr.write('\ttargeting affixes with smaller subword embeddings..\n')
        if masked_embeddings:
            sents = mask_low_freq(sents, wfs, n_freq_vocab=n_freq_vocab)
            stderr.write('\tmasking low frequency forms..\n')
    elif inductive_bias == 'semantic':
        window, min_n, max_n = 10, 4, 6
    else:
        raise Exception('Unknown inductive bias for learning FastText vectors: {}'.format(inductive_bias))
    # Train embeddings
    stderr.write('Learning {} FastText embeddings..\n'.format(inductive_bias))
    stderr.flush()
    ft_model = FastText(size=size, window=window, min_count=min_count, min_n=min_n, max_n=max_n)
    ft_model.build_vocab(sentences=sents)
    ft_model.train(model=model, sentences=sents, total_examples=len(sents), epochs=epochs)
    # Identify the vectors representing words we're modeling
    wf_matrix = []
    vector_2_wf = {}
    wf_dict = {}
    for wf in wfs:
        wf_dict[wf] = ft_model[wf]
        tup_vec = tuple(list(ft_model[wf]))
        assert tup_vec not in vector_2_wf
        vector_2_wf[tup_vec] = wf
        wf_matrix.append(ft_model[wf])
    return ft_model, wf_matrix, vector_2_wf, wf_dict


def custom_cluster(wf_matrix, n_clusters):

    # Check if we need to determine n_clusters or if it's given oracularly
    if n_clusters == 'blind':
        stderr.write('Determining number of cells with dispersion deceleration..\n')
        stderr.flush()
        n_clusters = get_n_clusters(np.array(wf_matrix))
        
    # Perform initial Kmeans clustering
    stderr.write('Clustering embeddings..\n')
    stderr.flush()
    kmeans = KMeans(n_clusters=n_clusters, n_jobs=10).fit(wf_matrix)
    clusters = dict((i, []) for i in range(len(kmeans.cluster_centers_)))
    cluster_sizes = Counter()
    for i in range(len(wf_matrix)):
        vector = wf_matrix[i]
        label = kmeans.labels_[i]
        clusters[label].append(vector)
        cluster_sizes[label] += 1
    assert len(clusters) == n_clusters == len(cluster_sizes)
    assert len(wf_matrix) == sum(len(clusters[label]) for label in clusters)

    # Rank clusters from largest to smallest
    ranked_centroid_labels = sorted(list(cluster_sizes), key=lambda x: len(clusters[x]), reverse=True)
    ranked_clusters = []
    ranked_centroids = []
    for label in ranked_centroid_labels:
        ranked_clusters.append(clusters[label])
        ranked_centroids.append(kmeans.cluster_centers_[label])
    eligible_centroids = list(ranked_centroids)

    return ranked_clusters, ranked_centroids, n_clusters


def get_n_clusters(data, n_runs=25, maxClusters=100):
    """n_clusters is chosen by taking the first k where the Reiman sum of the 2 previous cluster dispersion Reiman sums (deceleration) drops below the square root of the first measured deceleration"""

    disps = np.zeros(maxClusters)
    for idx, k in enumerate(range(1, maxClusters)):

        for _ in range(n_runs):
            km = KMeans(k)
            km.fit(data)
            
            disps[idx] += km.inertia_
        disps[idx] /= n_runs

        if k == 3:
            threshold = np.sqrt((disps[0] - disps[1]) - (disps[1] - disps[2]))
            stderr.write('\tDispersion deceleration threshold = {}\n'.format(round(threshold, 5)))
        if k > 3:
            decel = (disps[idx-2] - disps[idx-1]) - (disps[idx-1] - disps[idx])
            stderr.write('\tDispersion deceleration = {}\n'.format(round(decel, 5)))
            stderr.flush()
            if decel < threshold:
                stderr.write('\tLet K be {}\n'.format(k-1))
                stderr.flush()
                return k - 1
        
    return k


def mask_low_freq(sents, wfs, n_freq_vocab=250):
    most_frequent = Counter()
    for sent in sents:
        for token in sent:
            most_frequent[token] += 1
    most_frequent = dict((item[0], item[1]) for item in most_frequent.most_common(n_freq_vocab))
    for s_ind in range(len(sents)):
        for w_ind in range(len(sents[s_ind])):
            wf = sents[s_ind][w_ind]
            if wf not in wfs and wf not in most_frequent:
                sents[s_ind][w_ind] = '_'
    return sents

