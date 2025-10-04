# -*- coding: utf-8 -*-

from unittest import TestCase
from loda.ml.keras.program_generation_rnn import *

from loda.oeis import ProgramCache
from loda.ml import util
from tests.helpers import PROGRAMS_TEST_DIR


#class ProgramGenerationRNNTests(TestCase):
#
#    def setUp(self):
#        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)
#
#    def test_model(self):
#        model = train_model(self.program_cache)
#        model.save("test_model")
#        loaded = load_model("test_model")
#        loaded.summary()
#        generator = Generator(loaded, num_lanes=10)
#        for _ in range(10):
#            generator()


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
        program_ids = util.get_random_program_ids(self.program_cache)
        merged, _, _ = util.merge_programs(
            self.program_cache, program_ids=program_ids, num_ops_per_sample=3, num_nops_separator=3)
        return merged

    def test_program_to_tokens(self):
        merged = self.__merge_progs()
        tokens, vocabulary = util.program_to_tokens(merged)
        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(vocabulary), 0)
