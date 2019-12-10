from sys import stderr, stdout
import os
import tensorflow as tf
import numpy as np
import argparse
import multiprocessing as mp
import random
import math
import pickle as pkl
# Top level libraries
from grid import GridManager
from inflector import Inflector
from data_manager import DataManager
# Sub libraries
from Utils.matching import bipartite_match
from Utils.context_vectors import get_context_vectors
from Seq2seq import dataloader
from Seq2seq import model as model_lib


class Analyzer():

    def __init__(self, args, D, grids):
        pass
     
    def run_EM(self):
        pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2none(v):
    if v.lower() == 'none':
        return None
    else:
        return v


def get_arguments():

    parser = argparse.ArgumentParser()

    #############################################################################
    ## HIGH LEVEL STUFF
    #############################################################################
    parser.add_argument('-l', '--language', type=str, help='Language of data to analyze.', required=True)
    parser.add_argument('--restore_grid', default=None, type=str2none, help='None or the location of the pickled object containing the all grid objects to be restored.')
    parser.add_argument('--debug', default=True, type=str2bool, help='Prints a bunch of stuff like responsibility matrices after each M-step.')

    #############################################################################
    ### LOCATION OF INPUT DATA SETS AND OUTPUT FILES:
    #############################################################################
    parser.add_argument('-C', '--corpus', type=str, help='Location of the intrinsic evaluation corpus with one sentence per line. Tokens to be intrinsically evaluated should have "|||<lemma>|||<cell>" appended to them.', default=None)
    parser.add_argument('-L', '--lexicon', type=str, help='Location of the intrinsic evaluation lexicon with one realization per line. Lines are 3 columns, tab delimeted, containing the form, lemma, and then the cell.', default=None)
    parser.add_argument('-U', '--unimorph_intersect', type=str, help='location of an evaluation lexicon containing the full paradigms for any lemma in the intrinsic lexicon which also appears in UniMorph.')
    parser.add_argument('-e', '--extrinsic_analogies', type=str, default=None, help='location of an evaluation set of analogies. Each analogy is 4 words, the first 2 belonging to one paradigm, the second 2 belonging to another. The 1st and 3rd words realize the same cell in their respective paradigms, as do the 2nd and 4th. The first 3 words have all appeared in the lexicon, but crucially the 4th has not and must be predicted.')
    parser.add_argument('-m', '--model_location', type=str, help='Directory where the model outputs will be stored.', default='MyModel/model')

    #############################################################################
    ### GRID PARAMETERS
    #############################################################################
    parser.add_argument('-c', '--maybe_num_cells', type=str, help='If integer, set number of clusters to propose to k. If "blind", take the max number of forms per paradigm. If "oracle", take the gold number of cells per paradigm.', default='blind')
    parser.add_argument('-p', '--maybe_num_paradigms', type=str, help='If integer, set number of paradigms to p. If "blind", induce the number of paradigms empirically. If "oracle", take the gold number of paradigms.', default='blind')

    #############################################################################
    ## EM PARAMETERS
    #############################################################################
    parser.add_argument('-R', '--em_max_rounds', type=int, help='Number of rounds after which, EM will be terminated, regardless of convergence.', default=20)

    #############################################################################
    ## SEQ2SEQ PARAMETERS
    #############################################################################
    parser.add_argument('--s2s_max_dev_size', default=500, type=int, help='maximum number of instances to calculate dev loss on.')
    parser.add_argument('--s2s_data_format', default=dataloader.DataFormat.INFLECTION_IDX, type=dataloader.DataFormat,
        # choices=['INFLECTION', 'INFLECTION_IDX' , 'INFLECTION_IC', MT'],
        help='Data files should contain one instance per line. This flag defines the '
          'syntax of instances. The following options are supported: '
          'MT: 2 tab delimited fields. (1) space-delimited source sentence (2) '
          'space-delimited target sentence. '
          'INFLECTION: 3 tab deimited fields. (1) lemma as undelimited characters '
          '(2) target as undelimeted characters (3) morphosyntactic property set '
          'describing (2). This is the standard UniMorph format. '
          'INFLECTION_IC: 4 tab deimited fields. Like INFLECTION but with an extra field for inflection class.'
          )
    parser.add_argument('--s2s_model', default=model_lib.ModelFormat.TRANSFORMER, type=model_lib.ModelFormat,
    # choices=['TRANSFORMER'],
    help='Only one architecture is supported at present: '
      'TRANSFORMER: Soft, non-monotonic multi-head attention without recurrent '
      'cells. Input is assumed to be one-dimensional with features treated as '
      'additional sequence elements, as in the KANN_2016 set up.'
      )
    # Defining model hyperparameters.
    parser.add_argument('--s2s_max_num_epochs', default=100, type=int, help='Maximum number of epochs.')
    parser.add_argument('--s2s_patience', default=12, type=int,
    help='The number of epochs the accuracy on the '
    'dev set is allowed not to improve before training stops.')
    parser.add_argument('--s2s_batch_size', default=64, type=int, help='Size of batches fed to the model.')
    parser.add_argument('--s2s_val_batch_size', default=1000, type=int, help='Size of batches during validation.')
    parser.add_argument('--s2s_optimizer', default='adam', type=str, choices=['adam', 'adadelta'], help='Optimization algorithm.')
    parser.add_argument('--s2s_epsilon', default=0.000000001, type=float, help='Small constant for stability.')
    # Hyperparameters specific to transformer model.
    # Notation is kept consistent with this implementation:
    # https://www.tensorflow.org/beta/tutorials/text/transformer
    parser.add_argument('--s2s_num_layers', default=4, type=int, help='Number of encoder and decoder layers.')
    parser.add_argument('--s2s_d_model', default=128, type=int, help='Similar to num_units in the RNN models.')
    parser.add_argument('--s2s_num_heads', default=8, type=int, help='Number of attention heads.')
    parser.add_argument('--s2s_dff', default=512, type=int, help='Dimensions in the feed forward network in each individual encoder and decoder layer.')
    parser.add_argument('--s2s_dropout_rate', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--s2s_beta_1', default=0.9, type=float, help='Used for learning rate annealing.')
    parser.add_argument('--s2s_beta_2', default=0.98, type=float, help='Used for learning rate annealing.')
    parser.add_argument('--s2s_warmup_steps', default=4000, type=int, help='Learning rate will increase until this number of steps have been reached, then the learning rate will start annealing.')

    #############################################################################
    ### BENCHMARK VARIATIONS:
    #############################################################################
    parser.add_argument('--baseline', default=None, type=str2none, choices=[None, 'all_singletons', 'random', 'supervised_extrinsic', 'random_src'], help='Designations for different types of baselines.')
    parser.add_argument('--masked_embeddings', default=False, type=str2bool, help='Masks low frequency word forms when learning syntactic word embeddings before clustering them during the grid initialization.')
    parser.add_argument('--target_affix_embeddings', default=True, type=str2bool, help='Use smaller subword embeddings when learning syntactic word embeddings to motivate affix-based clustering during grid initialization.')
    parser.add_argument('--exponent_penalty', default=True, type=str2bool, help='Penalize long exponents when calculating scores of potential paradigms.')
    parser.add_argument('--exponent_penalty_discount', default=True, type=str2bool, help='Runs a second round initialization where exponent penalties are discounted by cell conditional exponent likelihoods.')
    parser.add_argument('--target_syntactic_windows', default=True, type=str2bool, help='Use a window size of 1 instead of 5 for syntactic embeddings to be clustered during grid initializations.')

    args = parser.parse_args()

    return args


#################################################################################
### MAIN
#################################################################################
if __name__ == '__main__':

    # Enable eager execution mode.
    tf.compat.v1.enable_eager_execution()

    # Parse input data
    args = get_arguments()
    D = DataManager(args)

    # Initialize analysis grid
    G = GridManager(D, args)
    D.c, D.p = G.c, min(x for x in (G.r, G.first_empty_row) if x)
    D.get_corpus_pot_lemmata(G)

    # Evaluate initialized grid intrinsically and extrinsically
    G.validate(eval_fn='intrinsic', msg='Intrinsic Grid Evaluation After Initialization')
    G.validate(eval_fn='extrinsic', msg='Extrinsic Analogical Grid Evaluation After Initialization')
    
    # Refine grid analyses
    raise Exception('TODO: Implement multi-task ANA.')
    ANA = Analyzer(args, D, G)
    ANA.run_EM()

    



