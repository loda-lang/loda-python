# -*- coding: utf-8 -*-

import os
import os.path
from loda.oeis import Sequence, ProgramCache

OEIS_TEST_DIR = os.path.join('tests', 'oeis')
OPERATIONS_TEST_DIR = os.path.join('tests', 'operations')
PROGRAMS_TEST_DIR = os.path.join('tests', 'programs', 'oeis')


def load_ops_params():
    files = os.listdir(OPERATIONS_TEST_DIR)
    ops = [(file[0:3],) for file in files]
    ops.sort()
    return ops


def load_programs_params():
    cache = ProgramCache(PROGRAMS_TEST_DIR)
    ids = cache.all_ids()
    return [(Sequence(id).id_str(), id,) for id in ids]
