# -*- coding: utf-8 -*-

import os
import os.path
from loda.oeis import Sequence

OPERATIONS_TEST_DIR = os.path.join('tests', 'operations')
PROGRAMS_TEST_DIR = os.path.join('tests', 'programs', 'oeis')


def load_ops_params():
    files = os.listdir(OPERATIONS_TEST_DIR)
    ops = [(file[0:3],) for file in files]
    ops.sort()
    return ops


def load_programs_params():
    ids = []
    for dir in os.listdir(PROGRAMS_TEST_DIR):
        for file in os.listdir(os.path.join(PROGRAMS_TEST_DIR, dir)):
            if file.startswith('A') and file.endswith('.asm'):
                ids.append(int(file[1:7]))
    ids.sort()
    return [(Sequence(id).id_str(), id,) for id in ids]
