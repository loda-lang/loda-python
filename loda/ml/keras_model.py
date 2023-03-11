from loda.lang import Program
from loda.oeis import ProgramCache
from .encoding import merge_programs, program_to_tokens, split_program, tokens_to_program

import tensorflow as tf
import os.path


class KerasModel(tf.keras.Model):

    num_ops_per_sample: int
    tokens: list[str]
    vocab: list[str]

    def __init__(self, program_cache: ProgramCache, num_ops_per_sample: int = 3,
                 batch_size: int = 16, buffer_size: int = 10000,
                 embedding_dim: int = 256, num_rnn_units: int = 1024):
        super().__init__(self)

        # Merge all programs into one program
        merged_programs = merge_programs(program_cache, num_ops_per_sample)
        self.num_ops_per_sample = num_ops_per_sample

        # Convert to tokens and vocabulary
        self.tokens, self.vocab = program_to_tokens(merged_programs)

        # Initialize TF lookup layers
        self.tokens_to_ids = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        self.ids_to_tokens = tf.keras.layers.StringLookup(
            vocabulary=self.tokens_to_ids.get_vocabulary(), invert=True, mask_token=None)

        # Convert the tokens to IDs
        self.ids = self.tokens_to_ids(self.tokens)

        # === Initialize datasets ===
        self.slice_dataset = tf.data.Dataset.from_tensor_slices(self.ids)
        # One operation is encoded using three tokens
        # plus one token because we split into input/output
        self.batch_dataset = self.slice_dataset.batch(
            3 * self.num_ops_per_sample + 1, drop_remainder=True)
        self.split_dataset = self.batch_dataset.map(self.__split_input_label)
        self.prefetch_dataset = (self.split_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

        # === Initialize layers ===
        self.vocab_size = len(self.tokens_to_ids.get_vocabulary())
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(num_rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def ids_to_tokens_str(self, ids) -> list[str]:
        return [t.numpy().decode("utf-8") for t in self.ids_to_tokens(ids)]

    def ids_to_programs(self, ids) -> list[Program]:
        return split_program(tokens_to_program(self.ids_to_tokens_str(ids)))

    def __split_input_label(self, sample: list):
        input = sample[:-1]
        label = sample[1:]
        return input, label

    def call(self, inputs, states=None, return_state=False, training=False):
        values = inputs
        values = self.embedding(values, training=training)
        if states is None:
            states = self.gru.get_initial_state(values)
        values, states = self.gru(
            values, initial_state=states, training=training)
        values = self.dense(values, training=training)
        if return_state:
            return values, states
        else:
            return values

    def fit_with_checkpoints(self, epochs: int, checkpoint_dir: str):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True)
        return self.fit(self.prefetch_dataset, epochs=epochs, callbacks=[checkpoint_callback])
