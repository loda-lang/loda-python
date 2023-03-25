"""
Keras RNN models for LODA program generation.

## Example

>>> # Train a model using existing programs:
>>> program_cache = ProgramCache("path/to/programs")
>>> model = train_model(program_cache)
>>>
>>> # Save a model to disk:
>>> model.save("sample_model")
>>>
>>> # Load a model from disk:
>>> model = load_model("sample_model")
>>>
>>> # Generated program using the model:
>>> generator = Generator(model)
>>> program = generator()
"""

import copy
import time

from loda.lang import Operation, Program
from loda.oeis import ProgramCache
from loda.ml import util

import tensorflow as tf


class Model(tf.keras.Model):
    """Keras model for program generation using RNN."""

    def __init__(self, vocabulary: list,
                 embedding_dim: int, num_rnn_units: int,
                 num_samples: int, sample_size: int,
                 num_ops_per_sample: int, num_nops_separator: int,
                 program_ids: list):

        super().__init__(self)
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.num_rnn_units = num_rnn_units
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.num_ops_per_sample = num_ops_per_sample
        self.num_nops_separator = num_nops_separator
        self.program_ids = program_ids

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
        return {"vocabulary": self.vocabulary,
                "embedding_dim": self.embedding_dim,
                "num_rnn_units": self.num_rnn_units,
                "num_samples": self.num_samples,
                "sample_size": self.sample_size,
                "num_ops_per_sample": self.num_ops_per_sample,
                "num_nops_separator": self.num_nops_separator,
                "program_ids": self.program_ids}

    def summary(self, line_length=None, positions=None, print_fn=None,
                expand_nested=False, show_trainable=False, layer_range=None):
        super().summary(line_length, positions, print_fn,
                        expand_nested, show_trainable, layer_range)
        print("Vocabulary size:", self.get_vocab_size())
        print("Sample size:", self.sample_size)
        print("Trained samples:", self.num_samples)
        print("Trained programs:", len(self.program_ids))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Generator:

    def __init__(self, model: Model, initial_program: Program = Program(), num_lanes: int = 1, temperature: float = 1.0):
        """
        Program generator based on a previously trained RNN model.

        Args:
            model: Previously trained or loaded `Model`.
            initial_program: Program to initialize the generation. This can be empty.
            num_lanes: Number of parallel lanes to use for program generation. Using more lanes
                potentially increases the program generation performance, but also the memory usage.
            temperature: Controls the randomness of the generated programs.
        """
        # Store members:
        self.model = model
        self.num_lanes = num_lanes
        self.__temperature = temperature
        # Prepare inputs and states:
        initial_program = self.__prepare_initial_program(initial_program)
        self.inputs = self.__program_to_input_ids(initial_program)
        self.states = None
        # Prepare lanes:
        self.token_lanes = []
        self.program_lanes = []
        for _ in range(self.num_lanes):
            self.token_lanes.append([])
            self.program_lanes.append(Program())
        # Prepare program queue:
        self.program_queue = []
        # Statistics:
        self.num_generated_programs = 0
        self.num_generated_tokens = 0
        self.num_generated_operations = 0
        self.num_generated_nops = 0
        self.num_token_errors = 0
        self.num_program_errors = 0
        self.start_time = time.time()

    def __call__(self) -> Program:
        """Generate a program."""
        while len(self.program_queue) == 0:
            self.__generate_programs()
        return self.program_queue.pop()

    def __ids_to_tokens_str(self, ids) -> list:
        return [t.numpy().decode("utf-8") for t in self.model.ids_to_tokens(ids)]

    def __prepare_initial_program(self, program: Program) -> Program:
        initial = copy.deepcopy(program)
        diff_sample_size = len(initial.operations) - \
            self.model.num_ops_per_sample
        if diff_sample_size > 0:
            initial.operations = initial.operations[diff_sample_size:]
        elif diff_sample_size < 0:
            tmp_program = Program()
            util.append_nops(tmp_program, -diff_sample_size)
            tmp_program.operations.extend(initial.operations)
            initial = tmp_program
        return initial

    def __program_to_input_ids(self, program: Program):
        tokens, _ = util.program_to_tokens(program)
        ids = self.model.tokens_to_ids(tokens).numpy()
        return tf.constant([ids] * self.num_lanes)

    def __generate_ids(self):

        # Execute the model.
        predicted_logits, states = self.model(inputs=self.inputs,
                                              states=self.states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.__temperature

        # Sample the output logits to generate token IDs.
        self.inputs = tf.random.categorical(predicted_logits, num_samples=1)
        self.states = states

    def __generate_tokens(self):
        self.__generate_ids()
        next_tokens = self.__ids_to_tokens_str(
            tf.squeeze(self.inputs, axis=-1))
        # print("TOKENS: {}".format(next_tokens))
        for i in range(self.num_lanes):
            self.token_lanes[i].append(next_tokens[i])
        self.num_generated_tokens += self.num_lanes

    def __generate_operations(self):
        # Generate three tokens for one operation:
        self.__generate_tokens()
        self.__generate_tokens()
        self.__generate_tokens()
        operations = []
        for i in range(self.num_lanes):
            op = util.tokens_to_operation(self.token_lanes[i], 0)
            while op is None:
                self.num_token_errors += 1
                self.token_lanes[i].pop(0)
                self.__generate_tokens()
                op = util.tokens_to_operation(self.token_lanes[i], 0)
            self.token_lanes[i].pop(0)
            self.token_lanes[i].pop(0)
            self.token_lanes[i].pop(0)
            operations.append(op)
        self.num_generated_operations += self.num_lanes
        return operations

    def __generate_programs(self):
        operations = self.__generate_operations()
        for i in range(self.num_lanes):
            if operations[i].type == Operation.Type.NOP:
                self.num_generated_nops += 1
                if len(self.program_lanes[i].operations) > 0:
                    try:
                        self.program_lanes[i].validate()
                        self.program_queue.append(self.program_lanes[i])
                    except Exception as e:
                        # print("PRORGRAM ERROR:", e)
                        # print(program_lanes[i])
                        self.num_program_errors += 1
                    self.program_lanes[i] = Program()
                    self.num_generated_programs += 1
            else:
                self.program_lanes[i].operations.append(operations[i])

    def get_stats_info_str(self) -> str:
        """
        Returns an info string containing useful stats about this generator including
        the number of generated programs, the generation speed, and error statistics.

        Example output:
        ```text
        generated programs: 233, speed: 17.43 programs/s, token errors: 0.03%, program errors: 6.01%, separator overhead: -0.40%
        ```
        """
        separator_overhead = 1 - (self.num_generated_nops /
                                  (self.num_generated_programs * self.model.num_nops_separator))
        return "generated programs: {}, speed: {:.2f} programs/s, token errors: {:.2f}%, program errors: {:.2f}%, separator overhead: {:.2f}%".format(
            self.num_generated_programs,
            self.num_generated_programs / (time.time() - self.start_time),
            100 * self.num_token_errors / self.num_generated_tokens,
            100 * self.num_program_errors / self.num_generated_programs,
            100 * separator_overhead)


def __create_dataset(ids: list, sample_size: int,
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


def load_model(model_path: str) -> Model:
    """
    Load a Keras RNN Model for program generation.

    The model should have been generated using `train_model` and saved before.

    Args:
        model_path: File system path to the model to be loaded.
    Return:
        Returns the loaded `Model`.
    """
    return tf.keras.models.load_model(model_path, custom_objects={"Model": Model})


def train_model(program_cache: ProgramCache, num_programs: int = -1,
                num_ops_per_sample: int = 32, num_nops_separator: int = 24,
                embedding_dim: int = 256, num_rnn_units: int = 1024,
                epochs: int = 3):
    """
    Train a Keras RNN model for program generation.

    Args:
        program_cache: Program cache that contains the programs used for training the model.
        num_programs: Number of programs used for training (-1 for all available programs).
        num_ops_per_sample: Number of operations per sample. We recommend to set this approximately
            to the length of the longest loops in the training programs. This enables the model
            to learn the structure of closed program loops and avoid generation of broken loops.
        num_nops_separator: Number of `nop` operations used as separator between trained programs.
            We recommend to set this to 75% of `num_ops_per_sample`, but at least 1.
        embedding_dim: Embedding dimensions.
        num_rnn_units: Number of RNN units.
        epochs: Number of epochs for training. 

    Return:
        This function returns the trained Keras model.
    """
    # Get random program IDs.
    program_ids = util.get_random_program_ids(program_cache, num_programs)

    # Load programs and convert to tokens and vocabulary.
    merged_programs, num_samples, sample_size = util.merge_programs(
        program_cache, program_ids,
        num_ops_per_sample=num_ops_per_sample,
        num_nops_separator=num_nops_separator)
    tokens, vocabulary = util.program_to_tokens(merged_programs)

    # Create Keras model and dataset, run the training, and save the model.
    program_ids = sorted(program_ids)
    model = Model(vocabulary,
                  embedding_dim=embedding_dim,
                  num_rnn_units=num_rnn_units,
                  num_samples=num_samples,
                  sample_size=sample_size,
                  num_ops_per_sample=num_ops_per_sample,
                  num_nops_separator=num_nops_separator,
                  program_ids=program_ids)
    ids = model.tokens_to_ids(tokens)
    dataset = __create_dataset(ids, sample_size=sample_size)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)
    model.fit(dataset, epochs=epochs)
    return model
