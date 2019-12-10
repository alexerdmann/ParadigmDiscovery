import random
from collections import Counter
import os


class DataManager:

    def __init__(self, args, vocabSize=250):
        self.lg = args.language
        self.gold = Lexicon(args.lexicon)
        self.wfs = set(self.gold.wf_2_lem)
        self.set_c(args.maybe_num_cells)
        self.set_p(args.maybe_num_paradigms)
        self.gold_UM_intersect = Lexicon(args.unimorph_intersect, allowable_cells=set(self.gold.cell_2_wf))
        self.ext_analogies = get_extrinsic_eval_data(args.extrinsic_analogies)
        self.corpus_tokens, self.corpus_types, self.corpus_frequent_types, self.sents, gold_sents = self.record_corpus(args.corpus, vocabSize=vocabSize)
        self.gold.add_sents(gold_sents)

    def set_c(self, c):
        if c == 'oracle':
            self.c = len(self.gold.cell_2_wf)
        elif c != 'blind':
            self.c = int(c)
        else:
            self.c = 'blind'

    def set_p(self, p):
        if p == 'oracle':
            self.p = len(self.gold.lem_2_wf)
        elif p != 'blind':
            self.p = int(p)
        else:
            self.p = 'blind'

    def get_corpus_pot_lemmata(self, grid, quit=False):
        try:
            self.corpus_possible_lemmata = []
            for wf in self.corpus_tokens:
                if wf in grid.wf_2_base:
                    self.corpus_possible_lemmata.append(grid.wf_2_base[wf])
                else:
                    self.corpus_possible_lemmata.append('OOV')
        except AttributeError:
            if quit:
                raise Exception('Something is wrong..')
            grid.get_analyses()
            self.get_corpus_pot_lemmata(grid, quit=True)

    def record_corpus(self, corpus, vocabSize=250):

        corpus_tokens = [] # corpus
        corpus_types = {}
        gold_sents = []
        sents = []
        with open(corpus) as open_corpus:
            for line in open_corpus:
                gold_sent = []
                sent = []
                for word in line.split():
                    wf = word
                    lemma, cell_unfactored = None, None
                    word = word.split('|||')
                    if len(word) > 2:
                        lemma, cell_unfactored = word[-2:]
                        lemma = '_'.join(lemma.split('_')[:-1])
                        wf = '|||'.join(word[:-2])
                    # Record all corpus forms
                    corpus_tokens.append(wf)
                    if wf in self.wfs and wf not in corpus_types:
                        corpus_types[wf] = len(corpus_types)
                    # Record gold tagging data
                    gold_sent.append((wf, lemma, cell_unfactored))
                    sent.append(wf)
                gold_sents.append(gold_sent)
                sents.append(sent)
        # Getting vocabulary to be used for context modeling
        wordCounts = Counter(corpus_tokens)
        corpus_frequent_types = { "<->" : 0 }
        for wi, ci in wordCounts.most_common(vocabSize):
            corpus_frequent_types[wi] = len(corpus_frequent_types)

        return corpus_tokens, corpus_types, corpus_frequent_types, sents, gold_sents


class Lexicon(object):

    def __init__(self, lexicon, allowable_cells=None):
        self.lem_2_wf, self.wf_2_lem, self.lem_2_cell_2_wf, self.cell_2_wf, self.wf_2_cell = {}, {}, {}, {}, {}
        with open(lexicon) as open_lexicon:
            self.extract_from_lex(open_lexicon, allowable_cells=allowable_cells)

    def extract_from_lex(self, open_lexicon, allowable_cells=None):

        self.lem_2_cell_2_wf = {}
        self.lem_2_wf = {}
        self.wf_2_lem = {}
        self.cell_2_wf = {}
        self.wf_2_cell = {}

        lines = []
        for line in open_lexicon:
            line = line.strip().split('\t')
            if len(line) == 3:
                lines.append(line)
        random.shuffle(lines)

        for [wf, lem, cell_unfactored] in lines:

            cell_unfactored = ';'.join(sorted(cell_unfactored.split(';')))
            if allowable_cells == None or cell_unfactored in allowable_cells:

                # Remove the POS tag preceding the lemma
                lem = '_'.join(lem.split('_')[:-1])

                # dicts for unsupervised training, i.e., no cells
                if wf not in self.wf_2_lem:
                    self.wf_2_lem[wf] = set()
                self.wf_2_lem[wf].add(lem)

                if lem not in self.lem_2_wf:
                    self.lem_2_wf[lem] = set()
                self.lem_2_wf[lem].add(wf)

                # dicts for gold evaluation, i.e., include cells
                if lem not in self.lem_2_cell_2_wf:
                    self.lem_2_cell_2_wf[lem] = {}
                if cell_unfactored not in self.lem_2_cell_2_wf[lem]:
                    self.lem_2_cell_2_wf[lem][cell_unfactored] = set()
                self.lem_2_cell_2_wf[lem][cell_unfactored].add(wf)

                if cell_unfactored not in self.cell_2_wf:
                    self.cell_2_wf[cell_unfactored] = set()
                self.cell_2_wf[cell_unfactored].add(wf)

                if wf not in self.wf_2_cell:
                    self.wf_2_cell[wf] = set()
                self.wf_2_cell[wf].add(cell_unfactored)

    def add_sents(self, sents):
        self.sents = sents


def get_extrinsic_eval_data(fn):
    ext_analogies = []
    with open(fn) as open_data:
        for line in open_data:
            analogy = line.strip().split('\t')
            assert len(analogy) == 4
            ext_analogies.append(analogy)
    return ext_analogies