import os.path

from loda.oeis import ProgramCache
from loda.runtime import Interpreter
from loda.ml import keras, util

import tensorflow as tf


class SampleLODA:

    def __init__(self):
        # Initialize LODA programs cache using *.asm files from tests folder
        programs_dir = os.path.join('tests', 'programs', 'oeis')
        # programs_dir = os.path.expanduser("~/loda/programs/oeis")
        self.program_cache = ProgramCache(programs_dir)

    def print_program(self):
        # Load the LODA program for the prime numbers (A000040.asm)
        # See also the integer sequence entry at https://oeis.org/A000040
        program = self.program_cache.get(40)  # numeric version of A000040
        print(program)

    def eval_program_to_seq(self):
        # Evaluate the program to an integer sequence
        program = self.program_cache.get(40)  # numeric version of A000040
        interpreter = Interpreter(program_cache=self.program_cache)
        sequence, _ = interpreter.eval_to_seq(program, num_terms=20)
        print("Evaluated to sequence: {}\n".format(sequence))

    def keras(self):

        # Configuration for training.
        num_programs = -1  # all programs
        num_ops_per_sample = 5

        # Load programs and convert to tokens and vocabulary.
        merged_programs, num_samples, sample_size = util.merge_programs(
            self.program_cache, num_programs=num_programs, num_ops_per_sample=num_ops_per_sample)
        tokens, vocabulary = util.program_to_tokens(merged_programs)

        # Create Keras model and convert tokens to IDs.
        model = keras.Model(vocabulary)
        print("Number of Samples:", num_samples)
        print("Sample Size:", sample_size)
        print("Vocabulary Size:  ", model.get_vocab_size())
        ids = model.tokens_to_ids(tokens)

        # Create Keras dataset and run the training.
        dataset = keras.create_dataset(ids, sample_size=sample_size)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss)
        model.fit(dataset, epochs=5)

        # Use the trained model to generate programs.
        generator = keras.Generator(model)
        num_lanes = 2
        print("Generated programs from trained model:")
        for p in generator.generate_programs(4, num_ops_per_sample=num_ops_per_sample, num_lanes=num_lanes):
            print(p)

        # Save the model to disk.
        model_file = "sample_model"
        model.save(model_file)

        # Load the model back from disk.
        loaded = keras.load_model(model_file)
        generator2 = keras.Generator(loaded)

        # Use the trained model to generate programs.
        print("Generated programs from loaded model:")
        for p in generator2.generate_programs(4, num_ops_per_sample=num_ops_per_sample, num_lanes=num_lanes):
            print(p)


if __name__ == "__main__":
    sample = SampleLODA()
    sample.print_program()
    sample.eval_program_to_seq()
    sample.keras()
