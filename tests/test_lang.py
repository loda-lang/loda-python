# -*- coding: utf-8 -*-

from parameterized import parameterized
from unittest import TestCase

from loda.lang import Program
from loda.oeis import ProgramCache
from tests.helpers import load_programs_params, PROGRAMS_TEST_DIR


class LangTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    @parameterized.expand(load_programs_params())
    def test_read_program(self, _, id):
        program_path = self.program_cache.path(id)
        with open(program_path, "r") as file:
            program_str = file.read()
            program = Program(program_str)
            self.assertEqual(program_str, str(program))
