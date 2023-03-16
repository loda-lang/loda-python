# -*- coding: utf-8 -*-

from unittest import TestCase

from loda.oeis import ProgramCache
from loda.ml import keras, util
from tests.helpers import PROGRAMS_TEST_DIR


class KerasTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)
        merged_programs, self.num_samples, _ = util.merge_programs(
            self.program_cache)
        self.tokens, self.vocabulary = util.program_to_tokens(merged_programs)

    def test_model_tokens_to_ids(self):
        model = keras.Model(self.vocabulary)
        ids = model.tokens_to_ids(self.tokens)
        self.assertGreater(len(ids), 0)
        self.assertEqual(len(self.tokens), len(ids))
        self.assertGreater(self.num_samples, 0)

    def test_generator_ids_to_programs(self):
        model = keras.Model(self.vocabulary)
        ids = model.tokens_to_ids(self.tokens)
        # We expect the same number of programs after decoding.
        num_programs = len(self.program_cache.all_ids())
        generator = keras.Generator(model)
        recovered_programs = generator.ids_to_programs(ids)
        self.assertEqual(num_programs, len(recovered_programs))


class UtilTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    def test_program_to_tokens_A000005(self):
        program = self.program_cache.get(5)
        tokens, vocab = util.program_to_tokens(program)
        # One operation is encoded using three tokens.
        self.assertEqual(3 * len(program.operations), len(tokens))
        self.assertEqual(16, len(vocab))
        program2 = util.tokens_to_program(tokens)
        self.assertEqual(program, program2)

    def test_merge_split(self):
        num_programs = len(self.program_cache.all_ids())
        merged, _, _ = util.merge_programs(self.program_cache)
        splitted = util.split_program(merged)
        self.assertEqual(num_programs, len(splitted))

    def test_program_to_tokens(self):
        merged, _, _ = util.merge_programs(self.program_cache)
        tokens, vocabulary = util.program_to_tokens(merged)
        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(vocabulary), 0)
