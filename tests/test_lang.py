# -*- coding: utf-8 -*-

import os.path
from parameterized import parameterized
from unittest import TestCase

from loda.lang import Program
from loda.oeis import ProgramCache
from tests.helpers import load_programs_params


class LangTests(TestCase):

    @parameterized.expand(load_programs_params())
    def test_read_program(self, _, id):
        program_cache = ProgramCache(os.path.join('tests', 'programs', 'oeis'))
        program_path = program_cache.path(id)
        with open(program_path, 'r') as file:
            program_str = file.read()
            program = Program(program_str)
            self.assertEqual(program_str, str(program))
