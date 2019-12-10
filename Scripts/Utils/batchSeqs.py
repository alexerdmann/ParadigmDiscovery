import sys
import random
import numpy as np

def batchByLength(seqs, batchBy=0, padDims=None, labelDims=None, maskDim=None, splitStr=False, batchSize=32, pad=None, verbose=False):
    if labelDims is None:
        labelDims = (-1,)

    nn = len(seqs[0])
    labelCvt = []
    for dim in labelDims:
        if dim < 0:
            labelCvt.append(nn + dim)
        else:
            labelCvt.append(dim)
    labelDims = labelCvt

    if padDims is None:
        padDims = range(min(labelDims))

    unpadDims = [xx for xx in range(nn)
                 if xx not in padDims and xx not in labelDims
                 and xx != maskDim]

    xDims = sorted(padDims + unpadDims)

    if verbose:
        print("Dimensions", padDims, "will be padded",
              unpadDims, "will be unpadded",
              labelDims, "will be in the label")
    
    if splitStr:
        measureBy = lambda xx: len(xx[batchBy].split())
        measure = lambda xx: len(xx.split())
    else:
        measureBy = lambda xx: xx[batchBy].shape[0]
        measure = lambda xx: xx.shape[0]

    if batchBy is not None:
        seqs.sort(key=measureBy)

    res = []
    
    for bi in range(0, len(seqs), batchSize):
        batch = seqs[bi : bi + batchSize]

        paddedCols = []
        labels = []

        for dim in xDims:
            if dim in padDims:
                ml = max([measure(xx[dim]) for xx in batch])
                # print("Padding dim", dim,
                #       "to length", ml, ":",
                #       [measure(xx[dim]) for xx in batch])
                padded = []

                for item in batch:
                    seq = item[dim]
                    if splitStr:
                        seq = seq.split()
                        padSeq = seq + [pad,] * (ml - len(seq))
                        assert(len(padSeq) == ml)
                    else:
                        padSeq = np.vstack(
                            [seq, np.tile(pad, (ml - seq.shape[0], 1))])
                        assert(padSeq.shape[0] == ml)
                    padded.append(padSeq)
                paddedCols.append(np.array(padded))

            else: #not padded
                col = []
                for item in batch:
                    seq = item[dim]
                    col.append(seq)
                paddedCols.append(np.array(col))

        if len(labelDims) == 1:
            for item in batch:
                label = item[labelDims[0]]
                labels.append(label)
            labels = np.array(labels)
        else:
            for dim in labelDims:
                labelCol = []
                for item in batch:
                    label = item[dim]
                    labelCol.append(label)
                labelCol = np.array(labelCol)
                labels.append(labelCol)

        if maskDim is None:
            res.append((paddedCols, labels))
        else:
            mask = []
            for item in batch:
                ms = item[maskDim]
                mask.append(ms)
            mask = np.array(mask)

            res.append((paddedCols, labels, mask))
        # print(len(paddedCols), "X cols", len(labels), "Y cols")
        # print([xi.shape for xi in paddedCols],
        #       [yi.shape for yi in labels])

    random.shuffle(res)
    return res

class Batcher:
    def __init__(self, lst):
        self.lst = lst
        self.ii = 0
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def next(self):
        res = self.lst[self.ii % len(self.lst)]
        self.ii += 1
        return res

class StreamingBatcher:
    def __init__(self, genExes, batchSize,
                 padDims, labelDims,
                 pad=None, maskDim=None, batchBy=0, test=False):
        self.gen = genExes
        self.pad = pad
        self.batchSize = batchSize
        self.batchBy = batchBy
        self.padDims = padDims
        self.labelDims = labelDims
        self.test = test
        self.bNum = 0
        self.maskDim = maskDim

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # print("Compiling batch", self.bNum)
        # self.bNum += 1
        batch = []
        for ii in range(self.batchSize):
            try:
                batch.append(next(self.gen))
            except StopIteration:
                if ii == 0 and self.maskDim:
                    # print("Warning: empty batch")
                    return tuple([ tuple(), tuple(), tuple()])
                else:
                    return tuple([ tuple(), tuple()])

        #print("Padding")

        batches = batchByLength(batch, padDims=self.padDims,
                                labelDims=self.labelDims,
                                maskDim=self.maskDim,
                                batchSize=self.batchSize,
                                batchBy=self.batchBy,
                                pad=self.pad,
                                verbose=False)
        if not self.test:
            return batches[0]
        else:
            # test mode: discard labels
            return batches[0][0]
