from loda.lang import Operation, Program
from loda.oeis import ProgramCache

from typing import Tuple
from random import shuffle
import tensorflow as tf


def encode_program(program: Program) -> Tuple[tf.RaggedTensor, list[str]]:
    tokens = []
    vocab = set()
    for op in program.ops:
        type = op.type.name.lower()
        target = str(op.target)
        source = str(op.source)
        tokens.append(type)
        tokens.append(target)
        tokens.append(source)
        vocab.add(type)
        vocab.add(target)
        vocab.add(source)
    return (tf.ragged.constant([tokens]), sorted(vocab))


def merge_programs(program_cache: ProgramCache) -> Program:
    # Get all program IDs
    ids = program_cache.all_ids()
    # Shuffle them
    shuffle(ids)
    # Merge the programs into one program
    merged = Program()
    for id in ids:
        program = program_cache.get(id)
        # We separate programs using nops
        merged.ops.append(Operation())  # nop
        for op in program.ops:
            if op.type != Operation.Type.NOP:
                merged.ops.append(op)
    return merged
