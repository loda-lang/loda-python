# -*- coding: utf-8 -*-

import os
from parameterized import parameterized
import sys
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from loda.oeis import getOeisProgramPath
from loda.program import Program


class ProgramTests(unittest.TestCase):
    """Program tests"""

    @parameterized.expand([(5,),(1075,)])
    def test_read(self, id):
        path = 'tests/programs/oeis/' + getOeisProgramPath(id)
        with open(path, 'r') as f:
            pstr = f.read()
            p = Program(pstr)
            self.assertEqual(pstr, str(p))


if __name__ == '__main__':
    unittest.main()
