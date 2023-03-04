# -*- coding: utf-8 -*-

from parameterized import parameterized
from unittest import TestCase

from loda.oeis import ProgramCache
from loda.ml import encode_program
from tests.helpers import load_programs_params, PROGRAMS_TEST_DIR


class MLTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    @parameterized.expand(load_programs_params())
    def test_encoding(self, _, id):
        program = self.program_cache.get(id)
        # TODO: test...
        encode_program(program)
