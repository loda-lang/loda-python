# -*- coding: utf-8 -*-

from unittest import TestCase

from loda.oeis import ProgramCache
from loda.ml import program_to_tokens, tokens_to_program, merge_programs, split_program
from tests.helpers import PROGRAMS_TEST_DIR


class MLTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    def test_program_to_tokens_A000005(self):
        program = self.program_cache.get(5)
        tokens, vocab = program_to_tokens(program)
        self.assertEqual(3*len(program.ops), len(tokens))
        self.assertEqual(16, len(vocab))
        program2 = tokens_to_program(tokens)
        self.assertEqual(program, program2)

    def test_merge_split(self):
        num_programs = len(self.program_cache.all_ids())
        merged = merge_programs(self.program_cache)
        splitted = split_program(merged)
        self.assertEqual(num_programs, len(splitted))
