# -*- coding: utf-8 -*-

from unittest import TestCase

from loda.oeis import ProgramCache
from loda.ml import keras, util
from tests.helpers import PROGRAMS_TEST_DIR


class KerasTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)
        self.num_ops_per_sample = 3
        self.num_nops_separator = 2
        merged_programs, self.num_samples, _ = util.merge_programs(
            self.program_cache,
            num_programs=-1,
            num_ops_per_sample=self.num_ops_per_sample,
            num_nops_separator=self.num_nops_separator)
        self.tokens, self.vocabulary = util.program_to_tokens(merged_programs)

    def test_model_tokens_to_ids(self):
        model = keras.Model(
            self.vocabulary, self.num_ops_per_sample, self.num_nops_separator)
        ids = model.tokens_to_ids(self.tokens)
        self.assertGreater(len(ids), 0)
        self.assertEqual(len(self.tokens), len(ids))
        self.assertGreater(self.num_samples, 0)


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

    def __merge_progs(self):
        merged, _, _ = util.merge_programs(
            self.program_cache, num_programs=-1, num_ops_per_sample=3, num_nops_separator=3)
        return merged

    def test_program_to_tokens(self):
        merged = self.__merge_progs()
        tokens, vocabulary = util.program_to_tokens(merged)
        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(vocabulary), 0)
