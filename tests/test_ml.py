# -*- coding: utf-8 -*-

from unittest import TestCase

from loda.oeis import ProgramCache
from loda.ml import encoding, KerasModel
from tests.helpers import PROGRAMS_TEST_DIR


class KerasModelTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)
        self.keras_model = KerasModel(self.program_cache)

    def test_model_data(self):
        # Check the number of programs.
        num_programs = len(self.program_cache.all_ids())
        self.assertGreater(num_programs, 0)

        # Check the number of operations of merged program.
        merged = encoding.merge_programs(
            self.program_cache, self.keras_model.num_ops_per_sample)
        num_operations = len(merged.operations)
        self.assertGreater(num_operations, 0)

        # Check the number of tokens in the model.
        # One operation is encoded using three tokens.
        self.assertEqual(3 * num_operations, len(self.keras_model.tokens))

        # Check that the vocabulary is non-empty.
        self.assertGreater(len(self.keras_model.vocab), 0)

        # We expect the same number of IDs as tokens.
        self.assertEqual(len(self.keras_model.tokens),
                         len(self.keras_model.ids))

        # We expect the same number of programs after decoding.
        programs = self.keras_model.ids_to_programs(self.keras_model.ids)
        self.assertEqual(num_programs, len(programs))


class EncodingTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    def test_program_to_tokens_A000005(self):
        program = self.program_cache.get(5)
        tokens, vocab = encoding.program_to_tokens(program)
        # One operation is encoded using three tokens
        self.assertEqual(3 * len(program.operations), len(tokens))
        self.assertEqual(16, len(vocab))
        program2 = encoding.tokens_to_program(tokens)
        self.assertEqual(program, program2)

    def test_merge_split(self):
        num_programs = len(self.program_cache.all_ids())
        merged = encoding.merge_programs(self.program_cache)
        splitted = encoding.split_program(merged)
        self.assertEqual(num_programs, len(splitted))
