# python3
"""Transformer model for morphological inflection."""


import enum
import sys
import numpy as np
import tensorflow as tf
from . import evaluation as eval_lib

TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


################################################################################
### Classes
################################################################################


class ModelFormat(enum.Enum):
    """Enum object describing data format and number of sources."""

    TRANSFORMER = 'TRANSFORMER'

    def uses_kann_style_features(self):
        return self in (ModelFormat.TRANSFORMER,)

    def is_transformer(self):
        return self in (ModelFormat.TRANSFORMER,)


class Model(object):
    """Interface for using transformer model for inflection."""

    def __init__(self, hparams, all_data, flags):
        """Initialize a model based on the transformer architecture.

        Args:
          hparams: TensorFlow hyperparameters.
          all_data: holds any relevant datasets and related objects
          needed to construct and use the model. For the transformer based model, we
          only use the following components of all_data..
            src_language_index) TensorFlow DataSet object containing an index for
            mapping between string and integerized representations.
            trg_language_index) Same as above but for the target language.
            dataset_train) TensorFlow dataset containing the batched training data.
            dataset_dev) Same as above but for dev.
            dataset_test) Same as above but for test.
            src_max_len_seq) Length of the longest source sequence in the train set
            or in the train set used to build the model we are restoring.
          flags: Command line arguments.
        """

        self.flags = flags
        self.hparams = hparams
        self.all_data = all_data
        self.src_language_index = all_data.src_language_index
        self.trg_language_index = all_data.trg_language_index
        self.dataset_train = self.all_data.dataset_train
        self.dev_srcs, self.dev_trgs = self.all_data.dataset_dev
        self.test_srcs, self.test_trgs = self.all_data.dataset_test
        self.max_len = self.all_data.src_max_len_seq
        self.best_checkpoint_path = None
        self.loss_object = None
        self.input_vocab_size = self.src_language_index.vocab_size + 2
        self.target_vocab_size = self.trg_language_index.vocab_size + 2
        self.dev_acc = 0
        self.test_acc = 0
        self.base_wf_tags_2_loss = {}

        # Set Optimizer and loss objects.
        self.set_optimizer()
        self.set_loss_objects()

        # Build Transformer model and checkpoint manager.
        self.transformer = Transformer(
            self.hparams.num_layers, self.hparams.d_model, self.hparams.num_heads,
            self.hparams.dff, self.input_vocab_size, self.target_vocab_size,
            self.hparams.dropout_rate)
        self.set_or_restore_checkpoint()

    def set_optimizer(self):
        learning_rate = CustomSchedule(self.hparams.d_model,
                                   warmup_steps=self.hparams.warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=self.hparams.beta_1, beta_2=self.hparams.beta_2,
            epsilon=self.hparams.epsilon)

    def set_loss_objects(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

    def set_or_restore_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(transformer=self.transformer,
                                              optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, self.hparams.checkpoint_dir, max_to_keep=1)
        # If checkpoint was provided at command line, restore it.
        if self.hparams.checkpoint_to_restore:
            sys.stderr.write(
              '\t\t\tRestoring model from checkpoint:\n\t\t\t{}\n'.format(
                  self.hparams.checkpoint_to_restore))
            sys.stderr.flush()
            self.checkpoint.restore(self.hparams.checkpoint_to_restore)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Compiles train_step into a TF graph for faster execution.
    @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
    def train_step(self, inp, trg):
        """Runs one batch of training as a graph-executable function."""

        trg_input = trg[:, :-1]
        trg_real = trg[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                         trg_input)
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, trg_input, True, enc_padding_mask,
                                        combined_mask, dec_padding_mask)
            loss = self.loss_function(trg_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                       self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(trg_real, predictions)

    def train(self):
        """Trains a transformer model."""

        sys.stderr.write('\t\t\tMax epochs: {}'.format(self.hparams.max_num_epochs))
        sys.stderr.write(', Max patience: {}'.format(self.hparams.patience))
        sys.stderr.write(', Warm up steps: {}\n'.format(self.hparams.warmup_steps))

        self.dev_acc = -1.0
        num_epochs_without_improvement = 0
        for epoch in range(self.hparams.max_num_epochs):

            sys.stderr.flush()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (_, (inp, trg)) in enumerate(self.dataset_train):

                self.train_step(inp, trg)

            dev_acc = self.validate()
            sys.stderr.write(
                '\t\t\tEpoch {} Loss {:.4f} Dev Acc {:.4f}\n'.format(
                    epoch + 1, self.train_loss.result(), dev_acc))

            if dev_acc > self.dev_acc:
                self.best_checkpoint_path = self.ckpt_manager.save()
                sys.stderr.write(
                    '\t\t\t\tSaving checkpoint for epoch {} at \n\t\t\t\t{}\n'.format(
                        epoch + 1, self.best_checkpoint_path))
                self.dev_acc = dev_acc
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if self.dev_acc > 0.0 and num_epochs_without_improvement >= self.hparams.patience:
                    sys.stderr.write(
                        '\t\t\tStopping early at epoch: {}\n'.format(epoch + 1))
                    break

        # Restore the best performing model.
        sys.stderr.write(
            '\t\t\t\tRestoring best checkpoint: \n\t\t\t\t{}\n'.format(self.best_checkpoint_path))
        self.checkpoint.restore(self.best_checkpoint_path)

        return self.best_checkpoint_path

    def validate(self, dev=True, best_checkpoint_path=None, predictions_file=None):
        """Validates a trained model on unseen sequences."""

        if dev:
            val_srcs, val_trgs = self.dev_srcs, self.dev_trgs
        else:
            val_srcs, val_trgs = self.test_srcs, self.test_trgs
        pred_seqs, src_seqs, trg_seqs = {}, {}, {}
        src_feat_bundles, trg_feat_bundles = {}, {}

        preds = self.translate(val_srcs)

        for val_id in range(len(val_srcs)):
            src = val_srcs[val_id]
            trg = val_trgs[val_id]
            pred = preds[val_id]
            pred_seqs[val_id], src_seqs[val_id], trg_seqs[val_id] = pred, src, trg
            # Transformer assumes all features are passed as part of the sequence.
            src_feat_bundles[val_id], trg_feat_bundles[val_id] = '', ''

        exact_match_accuracy = eval_lib.evaluate(pred_seqs, src_seqs, trg_seqs, src_feat_bundles,trg_feat_bundles, predictions_file=predictions_file)

        return exact_match_accuracy

    def translate(self, val_srcs):
        """Gets predictions and converts them from integerized form to strings."""

        predicted_sequences = []

        while val_srcs:
            val_srcs_batch = val_srcs[:self.hparams.val_batch_size]
            val_srcs = val_srcs[self.hparams.val_batch_size:]
            results, _ = self.evaluate(val_srcs_batch)

            # Convert batch results from integer to string space.
            for result in results:
                prediction = []
                result = result.numpy()[1:]
                for pred_idx in result:
                    if pred_idx <= len(self.trg_language_index.tokens):
                        prediction.append(self.trg_language_index.tokens[pred_idx - 1])
                    else:
                        break
                predicted_sequences.append(' '.join(prediction))

        return predicted_sequences

    def evaluate(self, inp_sequences):
        """Gets predictions and attention weights from a set of input sequences.

        Args:
          inp_sequences: List of sequences in character space. These sequences will
          contain SRC_* and TRG_* features as the transformer model, as implemented,
          requires these to represented as part of the input. The distribution of
          these features w.r.t. the normal sequence elements, i.e., characters, is
          determined by command line arguments.
        Returns:
          output: Matrix of predictions for each input sequence in integer space
          to be parsed by the translate function.
          attention_weights: Attention weight matrices for every input sequence.
        """

        batch_len = len(inp_sequences)

        # Integerize input sentences.
        encoded_inp_sequences = []
        for inp_sequence in inp_sequences:
            middle = []
            for ch in inp_sequence.split():
                try:
                    middle.append(self.src_language_index.tokens.index(ch) + 1)
                except ValueError:  # Handle OOV characters
                    # OOVs should be extremely rare because characters are closed class.
                    middle.append(0)
            encoded_inp_sequence = [self.src_language_index.vocab_size] + middle + [
                    self.src_language_index.vocab_size+1]
            encoded_inp_sequences.append(encoded_inp_sequence)
        # Pad encoder input.
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
                encoded_inp_sequences, maxlen=self.max_len, padding='post')
        # Convert encoder input to TF tensor.
        encoder_input = tf.convert_to_tensor(encoder_input)

        # Initialize decoder input.
        decoder_input = [[self.trg_language_index.vocab_size]]*batch_len
        output = tf.convert_to_tensor(decoder_input)

        # Start incrementally decoding.
        for _ in range(self.max_len):

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                    encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(
                    encoder_input, output, False, enc_padding_mask, combined_mask,
                    dec_padding_mask)

            # Select the last integerized sequence element from the seq_len dimension.
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # Concatentate predicted_id to output to get step i+1 decoder input.
            output = tf.concat([output, predicted_ids], axis=-1)

        return output, attention_weights

    def validate_forced(self, dev=False, predictions_file=None):
        """Gets losses by forcing each trg so we can determine the best trg for each src."""

        if dev:
            srcs = self.dev_srcs
            trgs = self.dev_trgs
        else:
            srcs = self.test_srcs
            trgs = self.test_trgs

        while srcs:
            srcs_batch_raw = srcs[:self.hparams.val_batch_size]
            srcs = srcs[self.hparams.val_batch_size:]
            trgs_batch_raw = trgs[:self.hparams.val_batch_size]
            trgs = trgs[self.hparams.val_batch_size:]

            # Convert srcs and trgs to integerized tensors.
            srcs_batch = self.prepare_for_forced_validation(srcs_batch_raw, self.src_language_index)
            trgs_batch = self.prepare_for_forced_validation(trgs_batch_raw, self.trg_language_index)
            assert len(srcs_batch) == len(trgs_batch)

            # Get losses.
            losses = self.forced_val_step(srcs_batch, trgs_batch)
            assert len(losses) == len(srcs_batch)

            for idx in range(len(srcs_batch)):

                s = srcs_batch_raw[idx].split()
                base = ''
                tag_tup = [None, None, None]
                for x in s:
                    if len(x) == 1:
                        base += x
                    if x.startswith('TRG_'):
                        tag_tup[0] = x
                    elif x.startswith('IC_'):
                        tag_tup[1] = x
                    elif x.startswith('Co_'):
                        tag_tup[2] = x
                    elif '_BASE' in x:
                        base += x.replace('_BASE', '_')
                tag_tup = tuple(tag_tup)
                wf = ''.join(trgs_batch_raw[idx].split())
                self.base_wf_tags_2_loss[(base, wf, tag_tup)] = losses[idx].numpy()
                if predictions_file:
                    predictions_file.write('{} \t{} \t-> \t{} \t with loss: {}\n'.format(base, str(tag_tup), wf, str(round(self.base_wf_tags_2_loss[(base, wf, tag_tup)], 5))))

        return self.base_wf_tags_2_loss

    def prepare_for_forced_validation(self, src_or_trg, integerizer):
        """Integerizes a batch of raw source or target representations."""

        encoded_sequences = []
        for sequence in src_or_trg:
            middle = []
            for ch in sequence.split():
                try:
                    middle.append(integerizer.tokens.index(ch) + 1)
                except ValueError:  # Handle OOV characters
                    # OOVs should be extremely rare because characters are closed class.
                    middle.append(0)
            encoded_sequence = [integerizer.vocab_size] + middle + [
                    integerizer.vocab_size+1]
            encoded_sequences.append(encoded_sequence)
        # Pad encoder input.
        src_or_trg_tensor = tf.keras.preprocessing.sequence.pad_sequences(
                encoded_sequences, maxlen=self.max_len, padding='post')
        # Convert encoder input to TF tensor.
        src_or_trg_tensor = tf.convert_to_tensor(src_or_trg_tensor)

        return src_or_trg_tensor

    def forced_val_step(self, inp, trg):
        """Gets losses for each input-target pair in the batch."""

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        (enc_padding_mask, combined_mask,
         dec_padding_mask) = create_masks(inp, trg_inp)

        predictions, _ = self.transformer(
            inp, trg_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        losses = self.forced_val_loss_function(trg_real, predictions)

        return losses

    def forced_val_loss_function(self, real, pred):

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_, 1)

class Transformer(tf.keras.Model):
    """Transformer Architecture."""

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, trg, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """Calls the transformer."""

        enc_output = self.encoder(inp, training, enc_padding_mask)
        # (batch_size, inp_seq_len, d_model)

        dec_output, attention_weights = self.decoder(
            trg, enc_output, training, look_ahead_mask, dec_padding_mask)
        # dec_output.shape == (batch_size, trg_seq_len, d_model)

        final_output = self.final_layer(dec_output)
        # (batch_size, trg_seq_len, target_vocab_size)

        return final_output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    """Architecture of a single layer within the encoder."""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Calls the encoder layer."""

        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """Architecture of a single layer within the decoder."""

    def __init__(self, d_model, num_heads, dff, rate=0.1):

        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output == (batch_size, input_seq_len, d_model)
        """Calls the decoder layer."""

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """Encoder architecture."""

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(self.d_model, num_heads, dff, rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Calls the Encoder."""

        seq_len = tf.shape(x)[1]

        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """Decoder architecture."""

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)

        self.dec_layers = [DecoderLayer(self.d_model, num_heads, dff, rate)
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Calls the decoder."""

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights
        # x.shape == (batch_size, target_seq_len, d_model)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-headed attention architecture."""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        """Splits x's last dimension into (num_heads, depth) and transpose."""

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A custom schedule for learning rate annealing."""

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self._d_model = d_model
        self._d_model = tf.cast(self._d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self._d_model) * tf.math.minimum(arg1, arg2)


################################################################################
### Functions
################################################################################


def create_masks(src, trg):
  """Creates all the masks used by the transformer.

  Args:
    src: Batched src.
    trg: Batched trg.
  Returns:
    enc_padding_mask: Negates padded cells in src.
    combined_mask: Negates cells that were padded in src or trg.
    dec_padding_mask: Negates padded cells in trg.
  """

  # Encoder padding mask.
  enc_padding_mask = create_padding_mask(src)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(src)

  # Used in the 1st attention block in the decoder.
  # Used to pad and mask future tokens in input received by decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(trg)[1])
  dec_target_padding_mask = create_padding_mask(trg)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # Add extra dimensions to pad the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
  # (..., seq_len_q, seq_len_k)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  denom = tf.math.sqrt(dk)
  scaled_attention_logits = scaled_attention_logits / denom

  # Add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # Softmax is normalized on last axis (seq_len_k) so scores will add to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  """Encodes sequence elements' positions relative to some current position."""

  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # Apply sin to even indices in the array; 2i.
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # Apply cos to odd indices in the array; 2i+1.
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)])
  # first dense layer: (batch_size, seq_len, dff)
  # second dense layer: (batch_size, seq_len, d_model)
