# -*- coding: utf-8 -*-

import os
import os.path
from parameterized import parameterized
from unittest import TestCase

from loda.lang import Operation
from loda.oeis import ProgramCache
from loda.runtime import calc_arith, Interpreter
from tests.helpers import load_ops_params, load_programs_params, OPERATIONS_TEST_DIR, PROGRAMS_TEST_DIR


class RuntimeTests(TestCase):

    @parameterized.expand(load_ops_params())
    def test_calc_arith(self, op):
        t = Operation.Type[op.upper()]
        path = os.path.join(OPERATIONS_TEST_DIR, op + '.csv')
        with open(path, 'r') as file:
            file.readline()  # skip header
            for line in file:
                k = line.strip().split(',')
                a, b, r = tuple(
                    map(lambda s: None if s == 'inf' else int(s), k))
                v = calc_arith(t, a, b)
                self.assertEqual(
                    r, v, "expected {}({},{})={}".format(op, a, b, r))

    @parameterized.expand(load_programs_params())
    def test_eval_to_seq(self, _, id):
        program_cache = ProgramCache(PROGRAMS_TEST_DIR)
        program = program_cache.get(id)
        seq_str = program.ops[1].comment.split(',')
        seq_expected = list(map(lambda v: int(v.strip()), seq_str))
        num_terms = len(seq_expected)
        self.assertTrue(num_terms >= 10)
        interpreter = Interpreter(program_cache=program_cache)
        seq, _ = interpreter.eval_to_seq(program, num_terms)
        self.assertEqual(seq_expected, seq)
