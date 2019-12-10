"""
Given sequential data in a specified format, we use the specified architecture to
predict target sequences from source sequences.

Example:
  python run_seq2seq.py --train ../ANA/Data/SIGMORPHON_2018/task1/all/ara-train-low --dev ../ANA/Data/SIGMORPHON_2018/task1/all/ara-dev --work_dir $PWD/DEV
"""

import sys
import os
import argparse
import tensorflow as tf
import dataloader
import model as model_lib
import seq2seq_runner


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
  # Determining work space and input/output files/formats.
  parser = argparse.ArgumentParser()

  parser.add_argument('--work_dir', default=None, type=str, help='Path to working directory.')
  parser.add_argument('--data_format', default=dataloader.DataFormat.INFLECTION_IDX, type=dataloader.DataFormat,
    # choices=['INFLECTION', 'MT'],
    help='Data files should contain one instance per line. This flag defines the '
      'syntax of instances. The following options are supported: '

      'MT: 2 tab delimited fields. (1) space-delimited source sentence (2) '
      'space-delimited target sentence. '

      'INFLECTION: 3 tab deimited fields. (1) lemma as undelimited characters '
      '(2) target as undelimeted characters (3) morphosyntactic property set '
      'describing (2). This is the standard UniMorph format. '
      ### ADD ADDITIONAL FORMATS FOR HANDLING IC AND CONTEXT EMBEDDINGS.
      )
  parser.add_argument('--train', default=None, type=str, help='Path to training data.')
  parser.add_argument('--dev', default=None, type=str, help='Path to dev data.')
  parser.add_argument('--test', default=None, type=str, help='Path to test data.')
  parser.add_argument('--model', default=model_lib.ModelFormat.TRANSFORMER, type=model_lib.ModelFormat,
    # choices=['TRANSFORMER'],
    help='Only one architecture is supported at present: '
      'TRANSFORMER: Soft, non-monotonic multi-head attention without recurrent '
      'cells. Input is assumed to be one-dimensional with features treated as '
      'additional sequence elements, as in the KANN_2016 set up.'
      )
  # Defining model hyperparameters.
  parser.add_argument('--max_num_epochs', default=100, type=int, help='Maximum number of epochs.')
  parser.add_argument('--patience', default=20, type=int,
    help='The number of epochs the accuracy on the '
    'dev set is allowed not to improve before training stops.')
  parser.add_argument('--batch_size', default=64, type=int, help='Size of batches fed to the model.')
  parser.add_argument('--val_batch_size', default=1000, type=int, help='Size of batches during validation.')
  parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'adadelta'], help='Optimization algorithm.')
  parser.add_argument('--epsilon', default=0.000000001, type=float, help='Small constant for stability.')

  # Hyperparameters specific to transformer model.
  # Notation is kept consistent with this implementation:
  # https://www.tensorflow.org/beta/tutorials/text/transformer
  parser.add_argument('--num_layers', default=4, type=int, help='Number of encoder and decoder layers.')
  parser.add_argument('--d_model', default=128, type=int, help='Similar to num_units in the RNN models.')
  parser.add_argument('--num_heads', default=8, type=int, help='Number of attention heads.')
  parser.add_argument('--dff', default=512, type=int, help='Dimensions in the feed forward network in each individual encoder and decoder layer.')
  parser.add_argument('--dropout_rate', default=0.1, type=float, help='Dropout rate.')
  parser.add_argument('--beta_1', default=0.9, type=float, help='Used for learning rate annealing.')
  parser.add_argument('--beta_2', default=0.98, type=float, help='Used for learning rate annealing.')
  parser.add_argument('--warmup_steps', default=4000, type=int, help='Learning rate will increase until this number of steps have been reached, then the learning rate will start annealing.')

  # Additional functionalities.
  parser.add_argument('--checkpoint_to_restore', default=None, type=str, help='Path to the checkpoint to restore. This can be used for continuing training or testing.')

  FLAGS = parser.parse_args()

  assert FLAGS.work_dir

  return FLAGS


def main():

  FLAGS = get_args()

  if FLAGS.train:
    if not FLAGS.dev:
      raise Exception('A dev set must be provided if training.')
  else:
    if not FLAGS.checkpoint_to_restore:
      raise Exception(
          'If not training, you must supply a checkpoint to restore a model.')
    if not FLAGS.dev or FLAGS.test:
      raise Exception(
          'If not training or evaluating, what are you even doing!?')

  model = seq2seq_runner.run(FLAGS)


if __name__ == '__main__':
  main()

