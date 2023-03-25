# -*- coding: utf-8 -*-

import os
import os.path
from parameterized import parameterized
from unittest import TestCase

from loda.lang import Operation
from loda.oeis import ProgramCache
from loda.runtime import exec_arithmetic, Evaluator, Interpreter
from tests.helpers import load_ops_params, load_programs_params, OPERATIONS_TEST_DIR, PROGRAMS_TEST_DIR


class RuntimeTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    @parameterized.expand(load_ops_params())
    def test_exec_arithmetic(self, op):
        t = Operation.Type[op.upper()]
        path = os.path.join(OPERATIONS_TEST_DIR, op + ".csv")
        with open(path, "r") as file:
            file.readline()  # skip header
            for line in file:
                k = line.strip().split(',')
                a, b, r = tuple(
                    map(lambda s: None if s == "inf" else int(s), k))
                v = exec_arithmetic(t, a, b)
                self.assertEqual(
                    r, v, "expected {}({},{})={}".format(op, a, b, r))

    @parameterized.expand(load_programs_params())
    def test_evaluator(self, _, id):
        program = self.program_cache.get(id)
        seq_str = program.operations[1].comment.split(",")
        seq_expected = list(map(lambda v: int(v.strip()), seq_str))
        num_terms = len(seq_expected)
        self.assertTrue(num_terms >= 10)
        interpreter = Interpreter(self.program_cache)
        evaluator = Evaluator(program, interpreter)
        seq = []
        for _ in range(num_terms):
            seq.append(evaluator())
        self.assertEqual(seq_expected, seq)
