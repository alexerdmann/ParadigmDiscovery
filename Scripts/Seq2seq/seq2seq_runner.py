# python3
"""Runs a single fold of training and evaluation."""


import sys
import os
import tensorflow as tf
from . import dataloader
from . import model as model_lib
from shutil import copyfile


class HParams:

  def __init__(self, flags, checkpoint_dir, checkpoint_prefix):
    self.max_num_epochs=flags.max_num_epochs
    self.patience=flags.patience
    self.batch_size=flags.batch_size
    self.val_batch_size=flags.val_batch_size
    self.epsilon=flags.epsilon
    self.optimizer=flags.optimizer
    self.num_batches=None
    self.checkpoint_dir=checkpoint_dir
    self.checkpoint_prefix=checkpoint_prefix
    self.checkpoint_to_restore=flags.checkpoint_to_restore
    self.d_model=flags.d_model
    self.num_heads=flags.num_heads
    self.dff=flags.dff
    self.dropout_rate=flags.dropout_rate
    self.beta_1=flags.beta_1
    self.beta_2=flags.beta_2
    self.warmup_steps=flags.warmup_steps
    self.num_layers=flags.num_layers


################################################################################
### Functions
################################################################################


def handle_preparation_flags(flags):
  """Establish work directory and initiate hparams from command line arguments.

  Args:
    flags: Command line arguments.
  Raises:
    Exception: If the work directory already exists to prevent overwriting it.
    NotImplementedError: If more than 3 splits are provided.
  Returns:
    hparams: hyperparameters.
    flags: updated flags.
  """

  # Establish work space.
  if os.path.exists(flags.work_dir):
    raise Exception(
        """Work directory already exists:\n\t{}
        Please delete this directory or specify a new one.""".format(
            flags.work_dir))
  os.mkdir(flags.work_dir)

  if flags.train:
    assert flags.dev
    checkpoint_dir = os.path.join(flags.work_dir, 'checkpoints')
    os.makedirs(checkpoint_dir)
  else:
    assert flags.checkpoint_to_restore
    checkpoint_dir = os.path.dirname(flags.checkpoint_to_restore)
  checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

  # Get hyperparameters.
  hparams = HParams(flags, checkpoint_dir, checkpoint_prefix)

  # Tell dataloader how the model wants to interpret morphological features.
  flags.data_format.set_feature_style(flags.model)

  return hparams, flags


def prepare_data(flags, hparams):
  """Prepares Data.

  Args:
    flags: Command line arguments.
    hparams: hyperparameters.

  Returns:
    dataloader.SplitObjects object.
  """

  # Get train/dev/test file objects ready for dataloader.
  data_files = {'train': None, 'dev': None, 'test': None}
  if flags.train:
    data_files['train'] = open(flags.train)
  if flags.dev:
    data_files['dev'] = open(flags.dev)
  if flags.test:
    data_files['test'] = open(flags.test)

  # Get train/dev/test compatible with a restored model.
  if hparams.checkpoint_to_restore:
    restore_vocab_map = os.path.join(os.path.dirname(hparams.checkpoint_to_restore), '../vocab_map.tsv')
    copyfile(restore_vocab_map,
                  os.path.join(flags.work_dir, 'vocab_map.tsv'))
    with open(os.path.join(flags.work_dir, 'vocab_map.tsv')
                              ) as vocab_map_file:
      # Get train/dev/test splits.
      split_objects = dataloader.get_splits(
          data_files, flags.data_format, vocab_map_file, hparams,
          restore=hparams.checkpoint_to_restore)

  # Get train/dev/test without restoring an old model.
  else:
    with open(os.path.join(flags.work_dir, 'vocab_map.tsv'),
                               'w') as vocab_map_file:
      # Get train/dev/test splits.
      split_objects = dataloader.get_splits(
          data_files, flags.data_format, vocab_map_file, hparams,
          restore=hparams.checkpoint_to_restore)

  # Close data files.
  for fn in data_files:
    if data_files[fn]:
      data_files[fn].close()

  hparams.num_batches = split_objects.num_batches

  if flags.model.is_transformer():
    if hparams.checkpoint_to_restore:
      restore_transformer_vocab_map = os.path.join(os.path.dirname(
          hparams.checkpoint_to_restore), '../transformer_vocab_map.tsv')
      copyfile(restore_transformer_vocab_map,
                    os.path.join(flags.work_dir, 'transformer_vocab_map.tsv'))
      with open(os.path.join(
          flags.work_dir, 'transformer_vocab_map.tsv')) as vocab_map_file:
        split_objects = dataloader.prepare_for_transformer(
            flags, hparams, split_objects, vocab_map_file)
    else:
      with open(os.path.join(
          flags.work_dir, 'transformer_vocab_map.tsv'), 'w') as vocab_map_file:
        split_objects = dataloader.prepare_for_transformer(
            flags, hparams, split_objects, vocab_map_file)

  return split_objects


def validate_held_out(pred_filename, model, best_checkpoint_path, dev=True, losses=False):
    """Validates model on dev or test.

    Args:
    pred_filename: Path to file to contain predictions made from dataset.
    model: Trained model.
    best_checkpoint_path: None or file prefix for checkpoint with best dev acc.
    If None, we take restore the last checkpoint instead of the best checkpoint.
    dev: If False, run on test set, use dev set otherwise.

    Returns:
    exact_match_accuracy: Exact match accuracy.
    """

    pred_file = open(pred_filename, 'w')

    if losses:
        _ = model.validate_forced(dev=dev, predictions_file=pred_file)
        to_return = model

    else:
        exact_match_accuracy = model.validate(
            dev=dev, best_checkpoint_path=best_checkpoint_path,
            predictions_file=pred_file)
        to_return = exact_match_accuracy

    pred_file.close()
    sys.stderr.write('\tPredictions located at {}\n'.format(
        pred_filename))
    sys.stderr.flush()

    return to_return

        


def write_out_results(results_filename, split_sizes, max_len_seq, max_len_ft, language_index, feature_index, exact_match_accuracy_dev, exact_match_accuracy_test, flags):
  """Writes out relevant statistics regarding training and evaluation.

  Args:
    results_filename: Output file location within working directory.
    split_sizes: Split sizes.
    max_len_seq: Longest sequence.
    max_len_ft: Largest feature bundle.
    language_index: Maps to/from integer space.
    feature_index: Maps to/from integer space.
    exact_match_accuracy_dev: Correct predictions / targets on dev.
    exact_match_accuracy_test: Exact match accuracy on test.
    flags: Command line arguments.
  """

  results_file = open(
      os.path.join(flags.work_dir, results_filename), 'w')
  results_file.write('Train size: {}\nDev size: {}\nTest size: {}\n\n'.format(
      split_sizes[0], split_sizes[1], split_sizes[2]))
  results_file.write('Longest sequence: {}\nLargest feature set: {}\n\n'.format(
      max_len_seq, max_len_ft))
  if language_index and feature_index:
    results_file.write('Vocabulary size: {}\nUnique features: {}\n\n'.format(
        len(language_index.vocab), len(feature_index.vocab)))

  if exact_match_accuracy_dev:
    exact_match_accuracy_dev_str = 'Dev Exact Match Accuracy: {}\n'.format(
        round(exact_match_accuracy_dev, 4))
    results_file.write(exact_match_accuracy_dev_str)
    sys.stderr.write('{}\n'.format(exact_match_accuracy_dev_str))

  if exact_match_accuracy_test:
    exact_match_accuracy_test_str = 'Test Exact Match Accuracy: {}\n'.format(
        round(exact_match_accuracy_test, 4))
    results_file.write(exact_match_accuracy_test_str)
    sys.stderr.write('{}\n'.format(exact_match_accuracy_test_str))

  sys.stderr.flush()

  results_file.close()


################################################################################
### Main
################################################################################


def run(flags, mode='normal'):
  """Trains and/or evaluates model on one fold."""

  get_losses = False
  if mode == 'ANA':
    get_losses = True

  # Initialize hparams from command line arguments.
  hparams, flags = handle_preparation_flags(flags)

  # Prepare data.
  all_data = prepare_data(flags, hparams)
  trg_language_index = all_data.trg_language_index
  trg_feature_index = all_data.trg_feature_index
  trg_max_len_seq = all_data.trg_max_len_seq
  trg_max_len_ft = all_data.trg_max_len_ft
  split_sizes = all_data.split_sizes

  # Get model.
  model = model_lib.Model(hparams, all_data, flags)

  if flags.train:
    # Train model and save checkpoints.
    best_checkpoint_path = model.train()
  else:
    best_checkpoint_path = hparams.checkpoint_to_restore

  if flags.dev and mode != 'ANA':
    # Validate on dev set.
    pred_filename = os.path.join(flags.work_dir, 'predictions_dev.txt')
    sys.stderr.write('Validating on Dev\n')
    sys.stderr.flush()
    model.dev_acc = validate_held_out(
        pred_filename, model, best_checkpoint_path)

  if flags.test:
    if get_losses:
        sys.stderr.write('\t\tdev acc: {}\n'.format(model.dev_acc))
        sys.stderr.write('\t\tCalculating losses for all base,wf tuples\n')
        sys.stderr.flush()
    # Validate on test set.
    pred_filename = os.path.join(flags.work_dir, 'predictions_test.txt')
    sys.stderr.write('Validating on Test\n')

    sys.stderr.flush()
    x = validate_held_out(
         pred_filename, model, best_checkpoint_path, dev=False, losses=get_losses)
    if not get_losses:
        model.test_acc = x  # otherwise, it's base_wf_tags_2_loss
    else:
        model = x

  # Save results for easy post-hoc analysis.
  if not get_losses:
      write_out_results('results.txt', split_sizes, trg_max_len_seq, trg_max_len_ft, trg_language_index, trg_feature_index, model.dev_acc, model.test_acc, flags)
      sys.stderr.write('Results located at {}\n'.format(
          os.path.join(flags.work_dir, 'results.txt')))
      sys.stderr.flush()

  return model
