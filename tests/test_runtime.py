# -*- coding: utf-8 -*-

from loda.lang import Operation
from loda.runtime import calc_arith
from parameterized import parameterized
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def load_ops_params():
    files = os.listdir('tests/operations')
    ops = [(f[0:3],) for f in files]
    ops.sort()
    return ops


class RuntimeTests(unittest.TestCase):
    """Runtime tests"""

    @parameterized.expand(load_ops_params())
    def test_calc_arith(self, f):
        t = Operation.Type[f.upper()]
        path = 'tests/operations/' + f + '.csv'
        with open(path, 'r') as file:
            file.readline()  # skip header
            for line in file:
                k = line.strip().split(',')
                a, b, r = tuple(
                    map(lambda s: None if s == 'inf' else int(s), k))
                v = calc_arith(t, a, b)
                self.assertEqual(
                    r, v, "expected {}({},{})={}".format(f, a, b, r))
