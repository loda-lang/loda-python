# -*- coding: utf-8 -*-

from unittest import TestCase

from loda.oeis import PrefixIndex, ProgramCache, Sequence
from tests.helpers import OEIS_TEST_DIR, PROGRAMS_TEST_DIR

NUM_SEQS = 5
NUM_PROGRAMS = 15


class PrefixIndexTests(TestCase):

    def setUp(self):
        self.index = PrefixIndex(OEIS_TEST_DIR)

    def test_index_size(self):
        self.assertEqual(NUM_SEQS, self.index.size())

    def test_index_get_A000004(self):
        a4: Sequence = self.index.get(4)
        self.assertEqual(4, a4.id)
        self.assertEqual("The zero sequence.", a4.name)
        self.assertEqual([0]*102, a4.terms)

    def test_index_get_A000005(self):
        a5: Sequence = self.index.get(5)
        self.assertEqual(5, a5.id)
        self.assertEqual(
            "d(n) (also called tau(n) or sigma_0(n)), the number of divisors of n.", a5.name)
        self.assertEqual([1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4,
                          4, 5, 2, 6, 2, 6, 4, 4, 2, 8, 3, 4, 4, 6,
                          2, 8, 2, 6, 4, 4, 4, 9, 2, 4, 4, 8, 2, 8,
                          2, 6, 6, 4, 2, 10, 3, 6, 4, 6, 2, 8, 4, 8,
                          4, 4, 2, 12, 2, 4, 6, 7, 4, 8, 2, 6, 4, 8,
                          2, 12, 2, 4, 6, 6, 4, 8, 2, 10, 5, 4, 2, 12,
                          4, 4, 4, 8, 2, 12, 4, 6, 4, 4, 4, 12, 2, 6,
                          6, 9, 2, 8, 2, 8], a5.terms)

    def test_global_match(self):
        m = self.index.global_match()
        expected = [i+1 for i in range(NUM_SEQS)]
        self.assertEqual(expected, self.index.get_match_ids(m))

    def test_refine_match_A000001(self):
        self.__test_refine([
            (0, [1, 4], True),
            (1, [1], True),
            (1, [1], True),
            (47, [], False),  # test incorrect term
        ])

    def test_refine_match_A000004(self):
        refinements = [(0, [1, 4], True)]
        refinements.extend([(0, [4], True)] * 100)
        refinements.append((0, [4], False))
        self.__test_refine(refinements)

    def test_refine_match_A000005(self):
        self.__test_refine([
            (1, [2, 3, 5], True),
            (2, [2, 5], True),
            (2, [2, 5], True),
            (3, [5], True),
            (2, [5], True),
            (47, [], False),  # test incorrect term
        ])

    def __test_refine(self, refinements):
        m = self.index.global_match()
        for (term, expected_ids, more) in refinements:
            self.assertEqual(more, self.index.refine_match(m, term))
            self.assertEqual(expected_ids, self.index.get_match_ids(m))


class ProgramCacheTests(TestCase):

    def setUp(self):
        self.program_cache = ProgramCache(PROGRAMS_TEST_DIR)

    def test_all_ids(self):
        num_programs = len(self.program_cache.all_ids())
        self.assertEqual(NUM_PROGRAMS, num_programs)
