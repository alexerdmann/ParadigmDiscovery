from sys import stderr, stdout
import os
import random
import numpy as np
from collections import defaultdict, Counter
from Utils.batchSeqs import batchByLength, Batcher, StreamingBatcher
from Utils.segment_functions import lcs, getExponent
import tensorflow as tf
import keras
import sklearn.cluster as skcluster
from keras.models import Model, load_model
from keras.constraints import UnitNorm
from keras.layers import Input, Dense, Multiply, Add,\
    TimeDistributed, Bidirectional, Activation, Lambda, \
    RepeatVector, Concatenate, dot, Reshape, GaussianNoise,\
    BatchNormalization
from keras.layers.embeddings import Embedding
from keras.activations import softmax
from keras.layers.recurrent import GRU
import keras.backend as K


def get_context_vectors(base_2_wf_2_cluster, cluster_2_base_2_wf, context_instances, context_vocab, nDim=20, panels=20, drc=False):

    nVocab = len(context_vocab) - 1
    
    stdout.write("\nTraining context vectors\n")
    stderr.write("\nTraining context vectors\n")
    cellNbrs = Counter()
    for inst in context_instances:
        (base, wf, posL, posR, 
         posAL, posAR, negL, negR, negAL, negAR) = inst
        for exes in [posL, posR]:
            for ex in exes:
                cellNbrs[ex] += 1

    stderr.write("\tBuilding network\n")
    model = build(nVocab, len(cluster_2_base_2_wf), panels, nDim, directional=drc)
    steps = (len(context_instances) * 2 * 2) // 32
    stderr.write("\t{} examples in lines out of {}\n".format(str(len(context_instances)), str(len(context_instances) * 2 * 2)))
    stderr.write("\tDecided to run {} batches\n".format(steps))

    for epoch in range(5):
        gen = streamingGen(context_instances, base_2_wf_2_cluster, context_vocab)
        gen = StreamingBatcher(gen, batchSize=32,
                               padDims=[],
                               labelDims=[2], batchBy=None)
        stdout.write("\tEpoch {}\n".format(epoch))
        model.fit_generator(gen, steps_per_epoch=steps, verbose=0)

    wf_embeddings = model.get_layer("contextEmbed")
    wf_params = wf_embeddings.get_weights()[0]

    cluster_embeddings = model.get_layer("cellEmbed")
    cluster_params = cluster_embeddings.get_weights()[0]

    wf_2_context_embedding = {}
    exponent_2_context_embedding = {}
    cluster_2_context_embedding = {}
    full_context_matrix = np.array([[0.0]*nDim]*(len(context_vocab)+len(cluster_2_base_2_wf)))
    wf_cluster_or_exp_2_embedding_row = {}

    mi = 0
    for wf in context_vocab:
        wf_embedding = wf_params[context_vocab[wf]]
        wf_cluster_or_exp_2_embedding_row[wf] = mi
        full_context_matrix[mi] = wf_embedding
        mi += 1

        if type(wf) == tuple:
            exponent_2_context_embedding[wf] = wf_embedding
        else:
            wf_2_context_embedding[wf] = wf_embedding
    for cluster in range(len(cluster_2_base_2_wf)):
        cluster_embedding = cluster_params[cluster]
        wf_cluster_or_exp_2_embedding_row[cluster] = mi
        full_context_matrix[mi] = cluster_embedding
        mi += 1
        cluster_2_context_embedding[cluster] = cluster_embedding


    return wf_2_context_embedding, exponent_2_context_embedding, cluster_2_context_embedding, full_context_matrix, wf_cluster_or_exp_2_embedding_row

def streamingGen(context_instances, base_2_wf_2_cluster, context_vocab, w2vDirectional=False):

    for (base, wf, posL, posR, posAL, posAR, 
         negL, negR, negAL, negAR) in context_instances:
        # print("EX", li, token, pos, neg)

        if w2vDirectional:
            if posAR[0] != "<->":
                posAR[0] = ">" + posAR[0]
                negAR[0] = ">" + negAR[0]

        # print("generating instance", token, "\t",
        #       posAL, posAR, "\tvs\t", negAL, negAR)

        for (wExe, aExe, direc, label) in ((posL, posAL, "left", 1), 
                                           (posR, posAR, "right", 1),
                                           (negL, negAL, "left", 0),
                                           (negR, negAR, "right", 0)):
            for ind in range(len(wExe)):
                w2vContext = np.zeros((1,))
                w2vLabel = np.zeros((1,))
                if wExe != "<->":
                    w2vLabel[0] = label
                w2vContext[0] = context_vocab[wExe[0]]
                if w2vDirectional:
                    if direc == "right":
                        w2vContext[0] += len(context_vocab) - 1

                affContext = np.zeros((1,))
                affLabel = np.zeros((1,))

                if aExe[0] != "<->":
                    affLabel[0] = label
                if aExe[0] in context_vocab:
                    affContext[0] = context_vocab[aExe[0]]
                else:
                    affContext[0] = context_vocab["<->"]

                cluster = random.choice(list(base_2_wf_2_cluster[base][wf]))
                #print("cell for", li, token, "->", cell)

                yield([cluster, w2vContext, w2vLabel])
                yield([cluster, affContext, affLabel])

def build(nVocab, nCell, panels, nEmbed, directional=False): #INPUTS: cell, exe, affexe // OUTPUTS: 0/1

    inp = Input((1,))
    cEmb = Embedding(
        nCell, nEmbed, name="cellEmbed",
    )(inp)
    cEmb = Lambda(lambda xx: K.squeeze(xx, 1))(cEmb)



    inpW2V = Input((1,))
    if directional:
        nW2V = (2 * nVocab) + 1
    else:
        nW2V = nVocab + 1

    contextEmb = Embedding(nW2V, 20, mask_zero=True,
                           name="contextEmbed",
    )(inpW2V)
    contextEmb = Lambda(lambda xx: K.squeeze(xx, 1))(contextEmb)
    score = Lambda(lambda xx:
                   K.sum(xx[0] * xx[1],
                         axis=-1, keepdims=True))([cEmb, contextEmb])
    a1 = Activation("sigmoid", name="w2vout1")(score)

    mdl = Model(inputs=[inp, inpW2V],
                outputs=[a1])
    # mdl.summary()
    mdl.compile(optimizer="rmsprop", #"adam",
                loss=["binary_crossentropy"],
                metrics={ 
                    "w2vout1" : "binary_accuracy",
                })

    return mdl                  


# if __name__ == "__main__":

#     get_context_vectors()