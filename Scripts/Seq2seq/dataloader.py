# python3
"""Classes and functions for loading data from specified types of corpora."""


import sys
import enum
import os
import typing
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from . import model as model_lib


class DataFormat(enum.Enum):
  """Describes data format and feature handling based on model architecture.

  Data format and model architecture are not entirely independent as some model
  architectures require morphological features to be interpreted as sequence
  elements while others maintain the capacity to handle features as
  separate entities. Thus instances of DataFormat must have set_feature_style()
  called in order to be valid. Set feature style peeks at the model architecture
  and interprets morphological features accordingly.
  """

  MT = 'MT'
  INFLECTION = 'inflection'
  INFLECTION_IDX = 'inflection_idx'
  INFLECTION_IC = 'inflection_ic'

  def is_oracular(self):  # i.e., IC is given
    return self in ()

  def is_inflection(self):
    return self in (DataFormat.INFLECTION, DataFormat.INFLECTION_IDX, DataFormat.INFLECTION_IC)

  def has_source_features(self):  # May not be necessary.
    return self in ()

  def set_feature_style(self, architecture):
    self.uses_kann_style_features = architecture in (model_lib.ModelFormat.TRANSFORMER,)


class ElementIndex(object):
  """Integerized index of an alphabet.

  Sequence alphabets and feature alphabets are integerized separately.
  """

  idx2element: typing.Dict[int, str]
  element2idx: typing.Dict[str, int]
  oov_idx: int
  pad_idx: int
  bos_idx: int
  eos_idx: int
  vocab: typing.List[str]

  def __init__(self, all_elements):
    self._all_elements = all_elements
    self.element2idx = {}
    self.idx2element = {}
    self.vocab_size = 0

    self._create_index()

  def _create_index(self):
    """Integerizes all unique members of _all_elements."""

    vocab_set = set()
    for element in self._all_elements:
      vocab_set.update([element])

    self.vocab = sorted(vocab_set)
    self.vocab_size = len(vocab_set) - 1  # Don't count OOV.
    if '<BOS>' in self.vocab:
      self.vocab.remove('<BOS>')
      self.vocab.append('<BOS>')
    if '<EOS>' in self.vocab:
      self.vocab.remove('<EOS>')
      self.vocab.append('<EOS>')
    self.vocab.append('<OOV>')

    self.element2idx['<PAD>'] = 0
    for idx, element in enumerate(self.vocab):
      self.element2idx[element] = idx + 1

    for element, idx in self.element2idx.items():
      self.idx2element[idx] = element

    self.oov_idx = self.element2idx['<OOV>']
    self.pad_idx = self.element2idx['<PAD>']
    self.bos_idx, self.eos_idx = None, None
    if '<BOS>' in self.element2idx:
      self.bos_idx = self.element2idx['<BOS>']
    if '<EOS>' in self.element2idx:
      self.eos_idx = self.element2idx['<EOS>']


class StringIntegerizer(object):
  """Transformer-friendly graph-executable integerizer of string inputs."""

  def __init__(self, src_language_index, trg_language_index):

    self.src_language_index = src_language_index
    self.trg_language_index = trg_language_index

  def encode(self, src, trg):
    """Encodes a string sequence as an integerized sequence.

    Args:
      src: Tensor of string elements of a source sequence, e.g., [SRC_Verb,
      SRC_PRS, SRC_PTPL, v, e, r, b, i, n, g, TRG_Verb, TRG_PST].
      trg: Tensor of string elements of a target sequence, e.g., [v, e, r, b, e,
      d].
    Returns:
      src: src but converted to an integerized list representation.
      trg: trg but converted to an integerized list representation.
    """

    # Necessary hacking due to weird unicode bugs in TensorFlow DataSet
    # tokenizer tool, e.g., using the TFDS tool's innate encode function removes
    # all diacritics in Arabic.
    src_middle = [self.src_language_index.tokens.index(
        ch) + 1 for ch in src.numpy().decode(
            'utf-8').split() if ch in self.src_language_index.tokens]
    trg_middle = [self.trg_language_index.tokens.index(
        ch) + 1 for ch in trg.numpy().decode(
            'utf-8').split() if ch in self.trg_language_index.tokens]

    src = [self.src_language_index.vocab_size] + src_middle + [
        self.src_language_index.vocab_size+1]
    trg = [self.trg_language_index.vocab_size] + trg_middle + [
        self.trg_language_index.vocab_size+1]

    return src, trg

  def tf_encode(self, src, trg):
    return tf.py_function(self.encode, [src, trg], [tf.int64, tf.int64])


class _PreprocessedInstance(typing.NamedTuple):
  """Named tuple describing a preprocessed instance.

  Named tuple containing:
  src: List of source sequence elements bookended by start and end tokens.
  trg: Same as src but for the target sequence.
  src_feats: List of categorical features relevant to src.
  trg_feats: List of categorical features relevant to trg.
  inflection_class: Inflection class categorical variable.
  lemma: The lemma.
  """

  src: typing.Optional[typing.List[str]]
  trg: typing.Optional[typing.List[str]]
  src_feats: typing.Optional[typing.List[str]]
  trg_feats: typing.Optional[typing.List[str]]
  inflection_class: typing.Optional[str]
  lemma: typing.Optional[str]


class _SplitTensors(typing.NamedTuple):
  """Named tuple describing all tensors relevant to a given data split.

  Named tuple containing:
  src_tensor_split: Split-specific tensor of sources.
  trg_tensor_split: Split-specific tensor of targets.
  src_feat_tensor_split: Split-specific tensor of source features.
  trg_feat_tensor_split: Split-specific tensor of target features.
  inflection_class_tensor_split: Split-specific tensor of inflection classes.
  lemmata_split: Split-specific tensor of lemmata.
  """

  src_tensor_split: typing.List[typing.List[typing.Union[typing.List[int],
                                                         int]]]
  trg_tensor_split: typing.List[typing.List[int]]
  src_feat_tensor_split: typing.List[typing.List[typing.Union[typing.List[int],
                                                              int]]]
  trg_feat_tensor_split: typing.List[typing.List[int]]
  inflection_class_tensor_split: typing.Any
  lemmata_split: typing.Optional[typing.List[str]]


class _DatasetObjects(typing.NamedTuple):
  """Named tuple describing all objects associated with a numpy dataset.

  Named tuple containing:
  src_tensor: Tensors covering all corpus sources.
  trg_tensor: Tensors covering all corpus targets.
  src_feat_tensor: Tensors covering all corpus source features.
  trg_feat_tensor: Tensors covering all corpus target features.
  inflection_class_tensor: Tensors covering all inflection classes instances.
  src_language_index: Maps to and from integer space in the source language.
  trg_language_index: Maps to and from integer space in the target language.
  src_feature_index: Maps to and from integer space for source features.
  trg_feature_index: Maps to and from integer space for target features.
  inflection_class_index: Maps to and from integer space for inflection classes.
  src_max_len_seq: Length of longest source sequence.
  trg_max_len_seq: Length of longest target sequence.
  src_max_len_ft: Length of largest source feature bundle.
  trg_max_len_ft: Length of largest target feature bundle.
  lemmata: List of all lemmata.
  split_sizes: Sizes of each split.
  """

  src_tensor: np.ndarray
  trg_tensor: np.ndarray
  src_feat_tensor: np.ndarray
  trg_feat_tensor: np.ndarray
  inflection_class_tensor: np.ndarray
  src_language_index: ElementIndex
  trg_language_index: ElementIndex
  src_feature_index: ElementIndex
  trg_feature_index: ElementIndex
  inflection_class_index: ElementIndex
  src_max_len_seq: int
  trg_max_len_seq: int
  src_max_len_ft: int
  trg_max_len_ft: int
  lemmata: typing.List[str]
  split_sizes: typing.List[int]


class _CorpusObjects(typing.NamedTuple):
  """Named tuple describing all corpus objects.

  Named tuple containing:
  corpus_srcs: List of all corpus sources.
  corpus_trgs: List of all corpus targets.
  corpus_src_feats: List of all corpus source features.
  corpus_trg_feats: List of all corpus target features.
  corpus_inflection_classes: List of all inflection classes instances.
  lemmata: List of all lemmata.
  split_sizes: Sizes of each split.
  src_language_index: Maps to and from integer space in the source language.
  trg_language_index: Maps to and from integer space in the target language.
  src_feature_index: Maps to and from integer space for source features.
  trg_feature_index: Maps to and from integer space for target features.
  inflection_class_index: Maps to and from integer space for inflection classes.
  """

  corpus_srcs: typing.Any
  corpus_trgs: typing.List[typing.List[int]]
  corpus_src_feats: typing.List[typing.List[typing.Union[typing.List[int],
                                                         int]]]
  corpus_trg_feats: typing.List[typing.List[int]]
  corpus_inflection_classes: typing.List[int]
  lemmata: typing.List[str]
  split_sizes: typing.List[int]
  src_language_index: ElementIndex
  trg_language_index: ElementIndex
  src_feature_index: ElementIndex
  trg_feature_index: ElementIndex
  inflection_class_index: ElementIndex


class _SplitObjects(typing.NamedTuple):
  """All objects returned by get_splits(...) function.

  Named tuple containing:
  src_language_index: Maps to and from integer space in the source language.
  trg_language_index: Maps to and from integer space in the target language.
  src_feature_index: Maps to and from integer space for source features.
  trg_feature_index: Maps to and from integer space for target features.
  inflection_class_index: Maps to and from integer space for inflection classes.
  src_max_len_seq: Length of longest source sequence.
  trg_max_len_seq: Length of longest target sequence.
  src_max_len_ft: Length of largest source feature bundle.
  trg_max_len_ft: Length of largest target feature bundle.
  split_sizes: Sizes of each split.
  dataset_train: TF train dataset.
  dataset_dev: TF dev dataset.
  dataset_test: TF test dataset.
  num_batches: Number of batches in the training set.
  lemmata_train: Lemmata for training.
  lemmata_dev: Lemmata for dev.
  lemmata_test: Lemmata for test.
  """

  src_language_index: ElementIndex
  trg_language_index: ElementIndex
  src_feature_index: ElementIndex
  trg_feature_index: ElementIndex
  inflection_class_index: ElementIndex
  src_max_len_seq: int
  trg_max_len_seq: int
  src_max_len_ft: int
  trg_max_len_ft: int
  split_sizes: typing.List[int]
  dataset_train: typing.Any
  dataset_dev: typing.Any
  dataset_test: typing.Any
  num_batches: int
  lemmata_train: typing.List[str]
  lemmata_dev: typing.List[str]
  lemmata_test: typing.List[str]


class _TFSplitObjects(typing.NamedTuple):
  """All objects returned by format_data_for_transformer(...) function.

  Named tuple containing:
  src_language_index: TensorFlow dataset object which maps to integer space.
  trg_language_index: TensorFlow dataset object which maps to integer space.
  src_feature_index: This is a placeholder for consistency with the RNN
  SplitObjects named tuple. But it will always be None here as the Transformer
  (as of now) does not distinguish features from elements, but encodes all
  features with the same index used for characters/language elements.
  trg_feature_index: See src_feature_index.
  src_max_len_seq: Length of longest source sequence.
  trg_max_len_seq: Length of longest target sequence.
  src_max_len_ft: Length of largest source feature bundle.
  trg_max_len_ft: Length of largest target feature bundle.
  split_sizes: Sizes of each split.
  dataset_train: TF train dataset.
  dataset_dev: TF dev dataset.
  dataset_test: TF test dataset.
  num_batches: Number of batches in the training set.
  integerizer: Transformer-friendly object enabling graph-executable
  integerization.
  """

  src_language_index: typing.Any
  trg_language_index: typing.Any
  src_feature_index: typing.Any
  trg_feature_index: typing.Any
  src_max_len_seq: int
  trg_max_len_seq: int
  src_max_len_ft: int
  trg_max_len_ft: int
  split_sizes: typing.List[int]
  dataset_train: typing.Any
  dataset_dev: typing.Any
  dataset_test: typing.Any
  num_batches: int
  integerizer: typing.Any


def accommodate_multi_word_inflections(inflection):
  return inflection.replace(' ', '_')

def can_be_int(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def preprocess_instance(data_format, instance) -> typing.Optional[_PreprocessedInstance]:
  """Preprocesses a train or test instance.

  Args:
    data_format: Format of each instance.
    instance: Format of seq model instances is infered from data_format.

  Returns:
    _PreprocessedInstance object.

  Raises:
    Exception: format specified by data_format is not supported (yet).
  """

  lemma = ''
  src_feats = []
  trg_feats = []
  inflection_class = ''
  instance = instance.strip()

  # Skip blank lines.
  if not instance:
    return None

  if data_format == DataFormat.MT:
    [src, trg] = instance.split('\t')
    src = src.split()
    trg = trg.split()

  elif data_format in (DataFormat.INFLECTION, DataFormat.INFLECTION_IDX):
    src, trg, trg_feats = instance.split('\t')
    lemma = src

  elif data_format == DataFormat.INFLECTION_IC:
    src, trg, trg_feats, inflection_class = instance.split('\t')
    lemma = src

  if data_format == DataFormat.MT:
    trg.insert(0, '<BOS>')
    trg.append('<EOS>')
    src.insert(0, '<BOS>')
    src.append('<EOS>')

  else:

    # target
    trg = list(accommodate_multi_word_inflections(trg))
    trg.insert(0, '<BOS>')
    trg.append('<EOS>')
    if trg_feats:
      trg_feats = trg_feats.split(';')
    # source
    if data_format == DataFormat.INFLECTION_IDX:
      # Include base marker
      src = ['{}_BASE'.format(src.split('_')[0])] + list(accommodate_multi_word_inflections(src.split('_', 1)[1]))
      # Deleter base marker
      # src = list(accommodate_multi_word_inflections(src.split('_', 1)[1]))
    else:
      src = accommodate_multi_word_inflections(src)
      src = list(src)
    if src_feats:
      src_feats = src_feats.split(';')
    src.insert(0, '<BOS>')
    if data_format.uses_kann_style_features:
      # Features will be treated as additional elements in the sequence.
      while src_feats:
        src.insert(1, 'SRC_{}'.format(src_feats.pop(-1)))
      src.extend(['TRG_{}'.format(t) for t in trg_feats])
      trg_feats = []
      if inflection_class:
        src.append('IC_{}'.format(inflection_class))
        inflection_class = ''
    src.append('<EOS>')    

  return _PreprocessedInstance(src, trg, src_feats, trg_feats, inflection_class, lemma)


def get_max_length(tensor):
  return max(len(t) for t in tensor)


def _get_index_from_corpus(corpus):
  """Gets a map to/from integer space for sequence elements or features.

  Args:
    corpus: A list of all corpus srcs, trgs, features, or inflection classes.

  Returns:
    element_index: Map to/from integer space for sequence, feature, or
    inflection class elements.

  Raises:
    Exception: If the corpus format is not interpretable.
  """
  # Get all elements.
  all_elements = []
  assert not isinstance(corpus, str)
  for instance in corpus:
    if isinstance(instance, str):  # Only for empty strings.
      all_elements.append(instance)
    else:
      for depth1 in instance:  # Covers all single-source instances.
        if isinstance(depth1, str):
          all_elements.append(depth1)
        else:
          raise Exception('There should not be any multi-source instances.')
          # for depth2 in depth1:  # Covers multi-source instances.
          #   if isinstance(depth2, str):
          #     all_elements.append(depth2)
          #   else:
          #     raise Exception('Instance is not incorrectly encoded: {}'.format(
          #         str(corpus)))

  # Integerize.
  element_index = ElementIndex(all_elements)

  # Run sanity checks.
  for idx in range(len(element_index.idx2element)):
    assert idx == element_index.element2idx[element_index.idx2element[idx]]
  assert (len(element_index.element2idx) == len(element_index.idx2element)
          == len(element_index.vocab) + 1)  # +1 b/c padding not in *.vocab.

  return element_index


def _get_tensor_from_corpus(corpus, element_index, is_feat=False):
  """Gets tensor with all src or trg sequences or all feature bundles in corpus.

  Args:
    corpus: Either corpus source sequences or corpus features as a numpy array.
    element_index: ElementIndex object.
    is_feat: Boolean that tells us not to sparsely encode the feature bundle.

  Returns:
    tensor: tensor with all src or trg sequences or feature bundles in corpus.
  """

  tensor = []
  for instance in corpus:
    if is_feat:
      tensor.append([element_index.pad_idx]*len(element_index.idx2element))
      for element in instance:
        idx = element_index.element2idx.get(element, element_index.oov_idx)
        tensor[-1][idx] = idx
    else:
      tensor.append(
          [element_index.element2idx.get(
              element, element_index.oov_idx) for element in instance])

  return tensor


def pad_tensor(tensor, pad_length):
  """Pads a tensor of any dimensionality along the last dimension.

  Args:
    tensor: Data tensor.
    pad_length: Length to pad to.

  Returns:
    tensor: Padded tensor of shape (... pad_length).
  """

  tensor = tf.keras.preprocessing.sequence.pad_sequences(
      tensor, maxlen=pad_length, padding='post'
      )

  return tensor


def _get_tensors_for_split(split_indices, lemmata, src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor, inflection_class_tensor, data_format) -> _SplitTensors:
  """Gets tensors by splits.

  Args:
    split_indices: Indices relevant to the split in question.
    lemmata: Plural of lemma.
    src_tensor: Tensor of all source sequences.
    trg_tensor: Tensor of all target sequences.
    src_feat_tensor: Tensor of all source feature bundles.
    trg_feat_tensor: Tensor of all target feature bundles.
    inflection_class_tensor: Tensor of all inflection classes.
    data_format: Format of data.

  Returns:
    _SplitTensor object.
  """

  # Build split-specific tensors.
  src_tensor_split, trg_tensor_split = [], []
  src_feat_tensor_split, trg_feat_tensor_split = [], []
  inflection_class_tensor_split, lemmata_split = [], []
  for idx in split_indices:
    lemmata_split.append(lemmata[idx])
    src_tensor_split.append(list(src_tensor[idx]))
    trg_tensor_split.append(list(trg_tensor[idx]))
    if data_format == DataFormat.MT:
      trg_feat_tensor_split.append([])
    else:
      trg_feat_tensor_split.append(list(trg_feat_tensor[idx]))
    if data_format.has_source_features():
      src_feat_tensor_split.append(list(src_feat_tensor[idx]))
    else:
      src_feat_tensor_split.append([])
    if data_format.is_oracular():
      inflection_class_tensor_split.append(list(inflection_class_tensor[idx]))
    else:
      inflection_class_tensor_split.append([])

  # Sanity Check.
  assert (len(src_tensor_split) == len(trg_tensor_split)
          == len(src_feat_tensor_split) == len(trg_feat_tensor_split)
          == len(inflection_class_tensor_split) == len(lemmata_split))

  return _SplitTensors(src_tensor_split, trg_tensor_split,
                       src_feat_tensor_split, trg_feat_tensor_split,
                       inflection_class_tensor_split, lemmata_split)


def load_restore_friendly_dataset(data_files, data_format, vocab_map_file) -> _DatasetObjects:
  """Prepares raw dataset for training compatible with a pre-trained model.

  Reads in a corpus. Integerizes sequence alphabets and attested features.
  Creates tensors for source and target sequences and all features from lists of
  all corpus instances, where each instance is a list of elements or features
  (depending on whether the tensor describes a sequence or a feature bundle) as
  represented by their corresponding integers. Computes longest sequence and
  largest feature bundle. Pads all vectors within each tensor to be as long as
  the longest of its type (sequence or feature).

  Args:
    data_files: FLAGS inherited from main file.
    data_format: Format of files in data_files.
    vocab_map_file: File object with integerized vocabulary map.

  Returns:
    _DatasetObjects object.
  """

  # Read in data and initialize integerized indices.
  corpus_objects = _read_data(data_files, data_format)
  corpus_srcs = corpus_objects.corpus_srcs
  corpus_trgs = corpus_objects.corpus_trgs
  corpus_src_feats = corpus_objects.corpus_src_feats
  corpus_trg_feats = corpus_objects.corpus_trg_feats
  corpus_inflection_classes = corpus_objects.corpus_inflection_classes
  lemmata = corpus_objects.lemmata
  split_sizes = corpus_objects.split_sizes
  src_language_index = corpus_objects.src_language_index
  trg_language_index = corpus_objects.trg_language_index
  src_feature_index = corpus_objects.src_feature_index
  trg_feature_index = corpus_objects.trg_feature_index
  inflection_class_index = corpus_objects.inflection_class_index

  # Read vocab map, overwrite existing index objects.
  src_language_index.vocab, src_feature_index.vocab = [], []
  src_language_index.element2idx, src_language_index.idx2element = {}, {}
  src_feature_index.element2idx, src_feature_index.idx2element = {}, {}
  src_max_len_seq, src_max_len_ft = 0, 0
  trg_language_index.vocab, trg_feature_index.vocab = [], []
  trg_language_index.element2idx, trg_language_index.idx2element = {}, {}
  trg_feature_index.element2idx, trg_feature_index.idx2element = {}, {}
  trg_max_len_seq, trg_max_len_ft = 0, 0
  inflection_class_index.vocab = []
  inflection_class_index.idx2element = {}
  inflection_class_index.element2idx = {}

  for line in vocab_map_file:
    line = line.strip().split('\t')
    if line:

      if 'src_max_len_seq' == line[0]:
        src_max_len_seq = int(line[1])
      elif 'src_max_len_ft' == line[0]:
        src_max_len_ft = int(line[1])
      elif 'src_vocab' == line[0]:
        idx, element = int(line[1]), line[2]
        src_language_index.idx2element[idx] = element
        src_language_index.element2idx[element] = idx
        if element != '<PAD>':
          src_language_index.vocab.append(element)
      elif 'src_feats' == line[0]:
        idx, element = int(line[1]), line[2]
        src_feature_index.idx2element[idx] = element
        src_feature_index.element2idx[element] = idx
        if element != '<PAD>':
          src_feature_index.vocab.append(element)

      if 'trg_max_len_seq' == line[0]:
        trg_max_len_seq = int(line[1])
      elif 'trg_max_len_ft' == line[0]:
        trg_max_len_ft = int(line[1])
      elif 'trg_vocab' == line[0]:
        idx, element = int(line[1]), line[2]
        trg_language_index.idx2element[idx] = element
        trg_language_index.element2idx[element] = idx
        if element != '<PAD>':
          trg_language_index.vocab.append(element)
      elif 'trg_feats' == line[0]:
        idx, element = int(line[1]), line[2]
        trg_feature_index.idx2element[idx] = element
        trg_feature_index.element2idx[element] = idx
        if element != '<PAD>':
          trg_feature_index.vocab.append(element)

      elif 'IC' == line[0]:
        idx, element = int(line[1]), ''
        if len(line) == 3:
          element = line[2]
        inflection_class_index.idx2element[idx] = element
        inflection_class_index.element2idx[element] = idx
        if element != '<PAD>':
          inflection_class_index.vocab.append(element)

  # Vectorize sequences.
  (src_tensor, trg_tensor, src_feat_tensor,
   trg_feat_tensor, inflection_class_tensor) = _vectorize_tensors(
       corpus_srcs, corpus_trgs, corpus_src_feats, corpus_trg_feats,
       corpus_inflection_classes, data_format, src_language_index,
       trg_language_index, src_feature_index, trg_feature_index,
       inflection_class_index)

  # Pad
  (src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
   inflection_class_tensor) = _pad_tensors(src_tensor, trg_tensor,
                                           src_feat_tensor, trg_feat_tensor,
                                           inflection_class_tensor, data_format,
                                           src_max_len_seq, trg_max_len_seq,
                                           src_max_len_ft, trg_max_len_ft)

  return _DatasetObjects(src_tensor, trg_tensor, src_feat_tensor,
                         trg_feat_tensor, inflection_class_tensor,
                         src_language_index, trg_language_index,
                         src_feature_index, trg_feature_index,
                         inflection_class_index, src_max_len_seq,
                         trg_max_len_seq, src_max_len_ft, trg_max_len_ft,
                         lemmata, split_sizes)


def _read_data(data_files, data_format) -> _CorpusObjects:
  """First step of load_(restore_friendly_)dataset."""

  # Read in data.
  (corpus_srcs, corpus_trgs, corpus_src_feats, corpus_trg_feats,
   corpus_inflection_classes, lemmata) = [], [], [], [], [], []
  split_sizes = [0, 0, 0]
  split_keys = ['train', 'dev', 'test']

  for split_idx in range(3):
    fn = data_files[split_keys[split_idx]]
    if fn:
      num_instances = 0
      for instance in fn:
        preprocessed_instance = preprocess_instance(data_format, instance)
        if preprocessed_instance:
          num_instances += 1
          corpus_srcs.append(preprocessed_instance.src)
          corpus_trgs.append(preprocessed_instance.trg)
          corpus_src_feats.append(preprocessed_instance.src_feats)
          corpus_trg_feats.append(preprocessed_instance.trg_feats)
          corpus_inflection_classes.append(
              preprocessed_instance.inflection_class)
          lemmata.append(preprocessed_instance.lemma)
      split_sizes[split_idx] = num_instances

  # Integerize data.
  all_sequences = corpus_srcs[:]
  all_sequences.extend(corpus_trgs)
  src_language_index = trg_language_index = _get_index_from_corpus(
      all_sequences)
  all_features = corpus_src_feats[:]
  all_features.extend(corpus_trg_feats[:])
  src_feature_index = trg_feature_index = _get_index_from_corpus(all_features)
  inflection_class_index = _get_index_from_corpus(corpus_inflection_classes)

  return _CorpusObjects(
      corpus_srcs, corpus_trgs, corpus_src_feats, corpus_trg_feats,
      corpus_inflection_classes, lemmata, split_sizes, src_language_index,
      trg_language_index, src_feature_index, trg_feature_index,
      inflection_class_index)


def _vectorize_tensors(corpus_srcs, corpus_trgs, corpus_src_feats, corpus_trg_feats, corpus_inflection_classes, data_format, src_language_index, trg_language_index, src_feature_index, trg_feature_index, inflection_class_index):
  """Vectorizes src, trg, and feat tensors."""

  src_tensor = _get_tensor_from_corpus(
      corpus_srcs, src_language_index)
  trg_tensor = _get_tensor_from_corpus(corpus_trgs, trg_language_index)
  src_feat_tensor = _get_tensor_from_corpus(
      corpus_src_feats, src_feature_index,
      is_feat=True)
  trg_feat_tensor = _get_tensor_from_corpus(
      corpus_trg_feats, trg_feature_index,
      is_feat=True)
  inflection_class_tensor = _get_tensor_from_corpus(
      corpus_inflection_classes, inflection_class_index)

  return (src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
          inflection_class_tensor)


def _pad_tensors(src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor, inflection_class_tensor, data_format, src_max_len_seq, trg_max_len_seq, src_max_len_ft, trg_max_len_ft):
  """Pads src, trg, and feat tensors."""

  src_tensor = pad_tensor(src_tensor,
                          src_max_len_seq)
  trg_tensor = pad_tensor(trg_tensor, trg_max_len_seq)
  src_feat_tensor = pad_tensor(src_feat_tensor,
                               src_max_len_ft)
  trg_feat_tensor = pad_tensor(trg_feat_tensor, trg_max_len_ft)
  inflection_class_tensor = pad_tensor(inflection_class_tensor, 1)

  return (src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
          inflection_class_tensor)


def load_dataset(data_files, data_format, vocab_map_file) -> _DatasetObjects:
  """Prepares raw dataset for training.

  Reads in a corpus. Integerizes sequence alphabets and attested features.
  Creates tensors for source and target sequences and all features from lists of
  all corpus instances, where each instance is a list of elements or features
  (depending on whether the tensor describes a sequence or a feature bundle) as
  represented by their corresponding integers. Computes longest sequence and
  largest feature bundle. Pads all vectors within each tensor to be as long as
  the longest of its type (sequence or feature).

  Args:
    data_files: FLAGS inherited from main file.
    data_format: Format of files in data_files.
    vocab_map_file: File object to contain integerized vocabulary map.

  Returns:
    _DatasetObjects object.
  """

  # Read in data and initialize integerized indices.
  corpus_objects = _read_data(data_files, data_format)
  corpus_srcs = corpus_objects.corpus_srcs
  corpus_trgs = corpus_objects.corpus_trgs
  corpus_src_feats = corpus_objects.corpus_src_feats
  corpus_trg_feats = corpus_objects.corpus_trg_feats
  corpus_inflection_classes = corpus_objects.corpus_inflection_classes
  lemmata = corpus_objects.lemmata
  split_sizes = corpus_objects.split_sizes
  src_language_index = corpus_objects.src_language_index
  trg_language_index = corpus_objects.trg_language_index
  src_feature_index = corpus_objects.src_feature_index
  trg_feature_index = corpus_objects.trg_feature_index
  inflection_class_index = corpus_objects.inflection_class_index

  # Vectorize sequences.
  (src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
   inflection_class_tensor) = _vectorize_tensors(
       corpus_srcs, corpus_trgs, corpus_src_feats, corpus_trg_feats,
       corpus_inflection_classes, data_format, src_language_index,
       trg_language_index, src_feature_index, trg_feature_index,
       inflection_class_index)

  # Get max tensor lengths.
  trg_max_len_seq = get_max_length(trg_tensor)
  trg_max_len_ft = get_max_length(trg_feat_tensor)
  src_max_len_seq = get_max_length(src_tensor)
  src_max_len_ft = get_max_length(src_feat_tensor)

  (src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
   inflection_class_tensor) = _pad_tensors(
       src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
       inflection_class_tensor, data_format, src_max_len_seq,
       trg_max_len_seq, src_max_len_ft, trg_max_len_ft)

  # Store Map.
  # Source language index.
  for idx in range(len(src_language_index.idx2element)):
    vocab_map_file.write('src_vocab\t{}\t{}\n'.format(idx,
        src_language_index.idx2element[idx]))
  vocab_map_file.write('src_max_len_seq\t{}\n'.format(src_max_len_seq))
  vocab_map_file.write('src_max_len_ft\t{}\n'.format(src_max_len_ft))
  # Source feature index.
  if src_feature_index:
    for idx in range(len(src_feature_index.idx2element)):
      vocab_map_file.write('src_feats\t{}\t{}\n'.format(idx,
          src_feature_index.idx2element[idx]))
  # Target language index.
  for idx in range(len(trg_language_index.idx2element)):
    vocab_map_file.write('trg_vocab\t{}\t{}\n'.format(idx,
        trg_language_index.idx2element[idx]))
  vocab_map_file.write('trg_max_len_seq\t{}\n'.format(trg_max_len_seq))
  vocab_map_file.write('trg_max_len_ft\t{}\n'.format(trg_max_len_ft))
  # Target feature index.
  if trg_feature_index:
    for idx in range(len(trg_feature_index.idx2element)):
      vocab_map_file.write('trg_feats\t{}\t{}\n'.format(idx,
          trg_feature_index.idx2element[idx]))
  # Inflection class index.
  if inflection_class_index:
    for idx in range(len(inflection_class_index.idx2element)):
      vocab_map_file.write('IC\t{}\t{}\n'.format(idx,
          inflection_class_index.idx2element[idx]))

  return _DatasetObjects(
      src_tensor, trg_tensor, src_feat_tensor, trg_feat_tensor,
      inflection_class_tensor, src_language_index, trg_language_index,
      src_feature_index, trg_feature_index, inflection_class_index,
      src_max_len_seq, trg_max_len_seq, src_max_len_ft, trg_max_len_ft, lemmata,
      split_sizes)


def get_splits(data_files, data_format, vocab_map_file, hparams, restore=None) -> _SplitObjects:
  """Loads specified data and divides it into specified train/dev/test splits.

  Args:
    data_files: File objects containing train/dev/test data.
    data_format: Format of files in data_files.
    vocab_map_file: File object for writing out intigerized vocab map.
    hparams: TensorFlow hyperparameters.
    restore: If restore, vocab_map_file should belong to a pre-trained model and
    it will be used to enforce consistent integerization of all data.

  Returns:
  _SplitObjects object.
  """

  # Load the full dataset.
  if restore:
    dataset_objects = load_restore_friendly_dataset(data_files, data_format,
                                                    vocab_map_file)
  else:
    dataset_objects = load_dataset(data_files, data_format, vocab_map_file)
  src_tensor = dataset_objects.src_tensor
  trg_tensor = dataset_objects.trg_tensor
  src_feat_tensor = dataset_objects.src_feat_tensor
  trg_feat_tensor = dataset_objects.trg_feat_tensor
  inflection_class_tensor = dataset_objects.inflection_class_tensor
  src_language_index = dataset_objects.src_language_index
  trg_language_index = dataset_objects.trg_language_index
  src_feature_index = dataset_objects.src_feature_index
  trg_feature_index = dataset_objects.trg_feature_index
  inflection_class_index = dataset_objects.inflection_class_index
  src_max_len_seq = dataset_objects.src_max_len_seq
  trg_max_len_seq = dataset_objects.trg_max_len_seq
  src_max_len_ft = dataset_objects.src_max_len_ft
  trg_max_len_ft = dataset_objects.trg_max_len_ft
  lemmata = dataset_objects.lemmata
  split_sizes = dataset_objects.split_sizes

  # Divide up indices.
  split_indices = [
      list(range(split_sizes[0])),
      list(range(split_sizes[0], sum(split_sizes[0:2]))),
      list(range(sum(split_sizes[0:2]), sum(split_sizes)))]

  # Build split-specific tensors.
  # Train.
  train_tensors = _get_tensors_for_split(
      split_indices[0], lemmata, src_tensor, trg_tensor, src_feat_tensor,
      trg_feat_tensor, inflection_class_tensor, data_format)
  src_tensor_train = train_tensors.src_tensor_split
  trg_tensor_train = train_tensors.trg_tensor_split
  src_feat_tensor_train = train_tensors.src_feat_tensor_split
  trg_feat_tensor_train = train_tensors.trg_feat_tensor_split
  inflection_class_tensor_train = train_tensors.inflection_class_tensor_split
  lemmata_train = train_tensors.lemmata_split
  # Dev.
  dev_tensors = _get_tensors_for_split(
      split_indices[1], lemmata, src_tensor, trg_tensor, src_feat_tensor,
      trg_feat_tensor, inflection_class_tensor, data_format)
  src_tensor_dev = dev_tensors.src_tensor_split
  trg_tensor_dev = dev_tensors.trg_tensor_split
  src_feat_tensor_dev = dev_tensors.src_feat_tensor_split
  trg_feat_tensor_dev = dev_tensors.trg_feat_tensor_split
  inflection_class_tensor_dev = dev_tensors.inflection_class_tensor_split
  lemmata_dev = dev_tensors.lemmata_split
  # Test.
  test_tensors = _get_tensors_for_split(
      split_indices[2], lemmata, src_tensor, trg_tensor, src_feat_tensor,
      trg_feat_tensor, inflection_class_tensor, data_format)
  src_tensor_test = test_tensors.src_tensor_split
  trg_tensor_test = test_tensors.trg_tensor_split
  src_feat_tensor_test = test_tensors.src_feat_tensor_split
  trg_feat_tensor_test = test_tensors.trg_feat_tensor_split
  inflection_class_tensor_test = test_tensors.inflection_class_tensor_split
  lemmata_test = test_tensors.lemmata_split

  # Convert tensors to numpy arrays.
  src_tensor_train, src_tensor_dev, src_tensor_test = np.array(
      src_tensor_train), np.array(src_tensor_dev), np.array(src_tensor_test)
  trg_tensor_train, trg_tensor_dev, trg_tensor_test = np.array(
      trg_tensor_train), np.array(trg_tensor_dev), np.array(trg_tensor_test)
  src_feat_tensor_train, src_feat_tensor_dev, src_feat_tensor_test = (
      np.array(src_feat_tensor_train), np.array(src_feat_tensor_dev),
      np.array(src_feat_tensor_test))
  trg_feat_tensor_train, trg_feat_tensor_dev, trg_feat_tensor_test = (
      np.array(trg_feat_tensor_train), np.array(trg_feat_tensor_dev),
      np.array(trg_feat_tensor_test))
  (inflection_class_tensor_train, inflection_class_tensor_dev,
   inflection_class_tensor_test) = (np.array(inflection_class_tensor_train),
                                    np.array(inflection_class_tensor_dev),
                                    np.array(inflection_class_tensor_test))

  # Define training parameters.
  buffer_size = len(src_tensor_train)
  num_batches = buffer_size // hparams.batch_size

  # Build TensorFlow datasets.
  # Train.

  dataset_train = tf.data.Dataset.from_tensor_slices(
      (src_tensor_train, trg_tensor_train, src_feat_tensor_train,
       trg_feat_tensor_train, inflection_class_tensor_train))
  if buffer_size:
    dataset_train = dataset_train.shuffle(buffer_size)
  dataset_train = dataset_train.batch(hparams.batch_size, drop_remainder=True)
  # Dev.
  dataset_dev = tf.data.Dataset.from_tensor_slices(
      (src_tensor_dev, trg_tensor_dev, src_feat_tensor_dev,
       trg_feat_tensor_dev, inflection_class_tensor_dev))
  dataset_dev = dataset_dev.batch(len(src_tensor_dev) + 1, drop_remainder=False)
  # Test.
  dataset_test = tf.data.Dataset.from_tensor_slices(
      (src_tensor_test, trg_tensor_test, src_feat_tensor_test,
       trg_feat_tensor_test, inflection_class_tensor_test))
  dataset_test = dataset_test.batch(
      len(src_tensor_test) + 1, drop_remainder=False)

  return _SplitObjects(
      src_language_index, trg_language_index, src_feature_index,
      trg_feature_index, inflection_class_index, src_max_len_seq,
      trg_max_len_seq, src_max_len_ft, trg_max_len_ft, split_sizes,
      dataset_train, dataset_dev, dataset_test, num_batches, lemmata_train,
      lemmata_dev, lemmata_test)


def prepare_for_transformer(flags, hparams, all_data, vocab_map_file) -> _TFSplitObjects:
  """Reformats _SplitObjects to suit the Transformer model.

  Args:
    flags: Command line arguments.
    hparams: Hyperparameters.
    all_data: _SplitObjects object.
    vocab_map_file: File object to either read a vocabulary from if restoring an
    old checkpoint or write a vocabulary to otherwise.
  Returns:
    _TFSplitObjects object.
  """

  dataset_train = all_data.dataset_train

  # Reformat inputs.
  (token_list, train_examples, buffer_size, max_len, dev_srcs, dev_trgs,
   test_srcs, test_trgs) = format_data_for_transformer(flags, hparams, all_data)

  # Update relevant objects given the Transformer-specific reformatting.
  (src_language_index, trg_language_index,
   max_len) = _get_transformer_language_index(
       hparams, vocab_map_file, token_list=token_list, max_len=max_len)
  integerizer = StringIntegerizer(src_language_index, trg_language_index)

  # Encode training data with updated language index.
  if flags.train:
    dataset_train = train_examples.map(integerizer.tf_encode)
    dataset_train = dataset_train.cache()  # To read in training data faster.
    dataset_train = dataset_train.shuffle(buffer_size).padded_batch(
        hparams.batch_size, padded_shapes=([-1], [-1]))
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

  return _TFSplitObjects(
      src_language_index, trg_language_index, src_feature_index=None,
      trg_feature_index=None, src_max_len_seq=max_len, trg_max_len_seq=max_len,
      src_max_len_ft=0, trg_max_len_ft=0, split_sizes=all_data.split_sizes,
      dataset_train=dataset_train, dataset_dev=(dev_srcs, dev_trgs),
      dataset_test=(test_srcs, test_trgs), num_batches=hparams.num_batches,
      integerizer=integerizer)


def _get_transformer_language_index(hparams, vocab_map_file, token_list=None, max_len=None):
  """Either restores tokenizers or learns them from provided vocabulary."""

  if hparams.checkpoint_to_restore:
    # Read src and trg tokenizer in from disc.
    restore_transformer_vocab_map_path = os.path.join(os.path.dirname(
        hparams.checkpoint_to_restore), '../transformer_vocab_map.tsv')
    sys.stderr.write('\t\t\tRestoring vocabulary map from \n\t\t\t{}\n'.format(
        restore_transformer_vocab_map_path))
    sys.stderr.flush()

    token_list = []
    for line in vocab_map_file:
      line = line.strip().split('\t')
      if line:
        if 'vocab' == line[0]:
          token_list.append(line[1])
        elif 'max_len' == line[0]:
          max_len = int(line[1])
      src_language_index = tfds.features.text.TokenTextEncoder(token_list)
      trg_language_index = tfds.features.text.TokenTextEncoder(token_list)

  else:
    # Get src and trg index and write to disc.
    assert isinstance(token_list, list)
    token_list.sort()
    src_language_index = tfds.features.text.TokenTextEncoder(token_list)
    trg_language_index = tfds.features.text.TokenTextEncoder(token_list)

    for idx in range(len(token_list)):
      vocab_map_file.write('vocab\t{}\n'.format(token_list[idx]))
    vocab_map_file.write('max_len\t{}\n'.format(max_len))

  return src_language_index, trg_language_index, max_len

def get_token_list(train_srcs, train_trgs, hparams, max_len_src):
  """Gets list of all tokens from training source and target sides combined."""
  token_list = {}
  max_len = 0
  for x_set in (train_srcs, train_trgs):
    for x_str in x_set:
      x_list = x_str.split()
      if not hparams.checkpoint_to_restore:
        max_len = max([max_len, len(x_list)])
      for token in x_list:
        token_list[token] = True

  # Source vector length can't be greater than vocab size, due to assumptions
  # made by TensorFlow to expedite training. This could be changed in the future
  # if it causes an issue, but for now, the extra vocabulary items provide no
  # confusion because the vocabulary size is tiny and these elements will not
  # be seen in training.
  novel_str = 'x'
  while len(token_list) < max_len_src:
    while novel_str in token_list:
      novel_str += 'x'
    token_list[novel_str] = True

  return list(token_list), max_len


def reformat_split_for_transformer(dataset, flags, all_data: _SplitObjects):
  """Reformats a single split to suit the Transformer model.

  Args:
    dataset: A TF dataset object for either train, dev, or test.
    flags: Command line arguments. Of particular importance is
    all_data: _SplitObjects object (pre- Transformer-specific reformatting)
  Returns:
    srcs: All srcs in this split after being flattened if necessary.
    trgs: All trgs in this split.
    max_len_srcs: Length of longest flattened source in this split.
  """

  srcs, trgs = [], []
  for (_, batch) in enumerate(dataset):
    src, trg = batch[0:2]
    src, trg = src.numpy(), trg.numpy()

    for i in range(len(src)):

      # Source.
      src_list = []
      for idx in src[i]:
        if idx in all_data.src_language_index.idx2element:
          if all_data.src_language_index.idx2element[idx] not in (
              '<PAD>', '<BOS>', '<EOS>'):
            src_list.append(all_data.src_language_index.idx2element.get(idx,
                                                                        ''))
      src_str = ' '.join(src_list)

      # Target.
      trg_list = []
      for idx in trg[i]:
        if idx in all_data.trg_language_index.idx2element:
          if all_data.trg_language_index.idx2element[idx] not in (
              '<PAD>', '<BOS>', '<EOS>'):
            trg_list.append(
                all_data.trg_language_index.idx2element.get(idx, ''))
      trg_str = ' '.join(trg_list)

      srcs.append(src_str)
      trgs.append(trg_str)

  return srcs, trgs, max([len(s.split()) for s in srcs])


def format_data_for_transformer(flags, hparams, all_data):
  """Reformats each split to enable graph-executable training of transformer."""

  dev_srcs, dev_trgs, test_srcs, test_trgs = [], [], [], []
  token_list = []
  max_len = 0
  # Build will complain about types if I don't do this:
  train_examples = tf.data.Dataset.from_tensor_slices(([], []))

  # Reformat train.
  buffer_size = 0
  if flags.train:
    token_list = {}
    (train_srcs, train_trgs,
     max_len_src) = reformat_split_for_transformer(all_data.dataset_train,
                                                   flags, all_data)

    if not hparams.checkpoint_to_restore:
      token_list, max_len = get_token_list(train_srcs, train_trgs, hparams,
                                           max_len_src)
    buffer_size = len(train_srcs)
    train_examples = tf.data.Dataset.from_tensor_slices(
        (train_srcs, train_trgs))

  # Reformat dev.
  if flags.dev:
    dev_srcs, dev_trgs, _ = reformat_split_for_transformer(
        all_data.dataset_dev, flags, all_data)

  # Reformat test.
  if flags.test:
    test_srcs, test_trgs, _ = reformat_split_for_transformer(
        all_data.dataset_test, flags, all_data)

  return (token_list, train_examples, buffer_size, max_len, dev_srcs, dev_trgs,
          test_srcs, test_trgs)

