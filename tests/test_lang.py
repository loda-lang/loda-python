# -*- coding: utf-8 -*-

from parameterized import parameterized
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from loda.lang import Program
from loda.oeis import getOeisProgramPath


class LangTests(unittest.TestCase):
    """LODA language tests"""

    @parameterized.expand([(5,),(1075,)])
    def test_read_program(self, id):
        path = 'tests/programs/oeis/' + getOeisProgramPath(id)
        with open(path, 'r') as f:
            pstr = f.read()
            p = Program(pstr)
            self.assertEqual(pstr, str(p))
