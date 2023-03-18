import os.path

from loda.lang import Operation, Program
from loda.ml import util

import tensorflow as tf
import os.path


class Model(tf.keras.Model):

    def __init__(self, vocabulary: list,
                 embedding_dim: int = 256, num_rnn_units: int = 1024):

        super().__init__(self)
        self.vocabulary = vocabulary

        # Initialize token <-> ID lookup layers.
        self.tokens_to_ids = tf.keras.layers.StringLookup(
            vocabulary=vocabulary, mask_token=None)
        self.ids_to_tokens = tf.keras.layers.StringLookup(
            vocabulary=self.tokens_to_ids.get_vocabulary(), invert=True, mask_token=None)
        vocab_size = self.get_vocab_size()

        # Create the processing layers.
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(num_rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def get_vocab_size(self):
        return len(self.tokens_to_ids.get_vocabulary())

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

    def get_config(self):
        return {"vocabulary": self.vocabulary}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def load_model(model_file):
    return tf.keras.models.load_model(model_file, custom_objects={"Model": Model})


class Generator:

    model: Model
    temperature: float

    def __init__(self, model: Model, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = model.tokens_to_ids(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[model.get_vocab_size()])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    def ids_to_tokens_str(self, ids) -> list:
        return [t.numpy().decode("utf-8") for t in self.model.ids_to_tokens(ids)]

    def ids_to_programs(self, ids) -> list:
        return util.split_program(util.tokens_to_program(self.ids_to_tokens_str(ids)))

    def program_to_input_ids(self, program: Program, num_lanes: int = 1):
        tokens, _ = util.program_to_tokens(program)
        ids = self.model.tokens_to_ids(tokens).numpy()
        return tf.constant([ids] * num_lanes)

    def generate_ids(self, inputs, states=None):

        # Execute the model.
        predicted_logits, states = self.model(inputs=inputs, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)

        # Return the prdicted IDs and model state.
        return predicted_ids, states

    def generate_tokens(self, inputs, states=None):
        next_ids, states = self.generate_ids(inputs, states=states)
        next_tokens = self.ids_to_tokens_str(tf.squeeze(next_ids, axis=-1))
        return next_ids, next_tokens, states

    def generate_operations(self, inputs, states=None):
        next_ids, types, states = self.generate_tokens(inputs, states)
        next_ids, targets, states = self.generate_tokens(next_ids, states)
        next_ids, sources, states = self.generate_tokens(next_ids, states)
        operations = []
        for i in range(len(types)):
            operations.append(util.tokens_to_operation(
                types[i], targets[i], sources[i]))
        return next_ids, operations, states

    def generate_programs(self, num_programs: int, num_ops_per_sample: int, num_lanes: int = 10):
        states = None
        initial = Program()
        util.append_nops(initial, num_ops_per_sample)
        next_ids = self.program_to_input_ids(
            initial, num_lanes=num_lanes)
        lanes = [Program()] * num_lanes
        programs = []
        while len(programs) < num_programs:
            next_ids, operations, states = self.generate_operations(
                next_ids, states=states)
            for i in range(num_lanes):
                if operations[i].type == Operation.Type.NOP:
                    if len(lanes[i].operations) > 0:
                        programs.append(lanes[i])
                        lanes[i] = Program()
                else:
                    lanes[i].operations.append(operations[i])
        return programs


def create_dataset(ids: list, sample_size: int,
                   batch_size: int = 128, buffer_size: int = 10000):

    # Basic tensor dataset.
    slice_dataset = tf.data.Dataset.from_tensor_slices(ids)

    # We repeat the original dataset to make sure we sample at all
    # possible start positions. We made sure already before that the
    # the dataset size mod the sample size is +/-1. So this works!
    # Note also that we don't need to enable drop_remainder here.
    batch_dataset = slice_dataset.repeat(sample_size).batch(sample_size)

    # Split the samples into (input,label) pairs.
    split_dataset = batch_dataset.map(util.split_sample)

    # Shuffle dataset.
    prefetch_dataset = (split_dataset.shuffle(buffer_size).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE))

    return prefetch_dataset
