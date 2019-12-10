# python3
"""Functions for evaluating sequence model."""


def write_out_prediction(predictions_file, src_seqs,
                         trg_seqs, pred_string, src_feat_bundles,
                         trg_feat_bundles, val_id):
  """Write out a single prediction instance to a text file.

  Args:
    predictions_file: Writeable file object.
    src_seqs: Dictionary of all (potentially nested) source instances.
    trg_seqs: Dictionary of all target instances.
    pred_string: Prediction as space delimited string of characters.
    src_feat_bundles: Dictionary of all (nested) src feature instances.
    trg_feat_bundles: Dictionary of all (nested) trg feature instances.
    val_id: The validation set index is also the key to all above dictionaries.
  """

  output_lines = []
  if trg_seqs[val_id] != pred_string:
    output_lines.append('*ERROR*')
  output_lines.append('SRC: {}'.format(src_seqs[val_id]))
  if src_feat_bundles[val_id]:
    output_lines.append('SFT: {}'.format(src_feat_bundles[val_id]))
  if trg_feat_bundles[val_id]:
    output_lines.append('TFT: {}'.format(trg_feat_bundles[val_id]))
  output_lines.append('TRG: {}'.format(trg_seqs[val_id]))
  output_lines.append('PRD: {}\n'.format(pred_string))
  predictions_file.write('{}\n'.format('\n'.join(output_lines)))


def evaluate(pred_seqs, src_seqs, trg_seqs, src_feat_bundles, trg_feat_bundles,
             predictions_file=None):
  """Computes evaluation metrics.

  Args:
    pred_seqs: Dictionary of predicted strings given relevant source instance.
    src_seqs: Dictionary of all (potentially nested) source instances.
    trg_seqs: Dictionary of all target instances.
    src_feat_bundles: Dictionary of all (nested) source feature instances. In
    case the model doesn't distinguish between sequence and feature elements, we
    will construct src_feat_bundles by parsing features in src_seqs.
    trg_feat_bundles: Dictionary of all (nested) target feature instances. In
    case the model doesn't distinguish between sequence and feature elements, we
    will construct trg_feat_bundles by parsing features in trg_seqs.
    predictions_file: Writeable file object or None.
  Returns:
    exact_match_accuracy: The exact match accuracy.
  """

  num_correct = 0
  for val_id in range(len(pred_seqs)):
    pred_string = ' '.join(pred_seqs[val_id].split())

    # Parse flat single src input into features and src infelction.
    src_form = ' '.join(
        [s for s in src_seqs[val_id].split() if len(s) == 1])
    src_feats = ' '.join(
        [s for s in src_seqs[val_id].split() if s.startswith('SRC_')])
    if not src_feats:
      src_feats = src_feat_bundles[val_id]
    trg_feats = ' '.join(
        [t for t in src_seqs[val_id].split() if t.startswith('TRG_')])
    if not trg_feats:
      trg_feats = trg_feat_bundles[val_id]
    if isinstance(src_feats, list):
      src_feats = ' '.join(src_feats)
    # A source is unique if the tuple of its form and features is unique.
    src_key = (src_form, src_feats)
    trg_key = (trg_seqs[val_id], trg_feats)

    # Mark correct predictions.
    if pred_string == trg_seqs[val_id]:
      num_correct += 1

    # Write out predictions.
    if predictions_file:
      write_out_prediction(predictions_file, src_seqs,
                           trg_seqs, pred_string, src_feat_bundles,
                           trg_feat_bundles, val_id)

  exact_match_accuracy = 100 * (num_correct / len(pred_seqs))

  return exact_match_accuracy
