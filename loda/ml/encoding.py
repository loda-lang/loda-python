from loda.lang import Operand, Operation, Program
from loda.oeis import ProgramCache

from typing import Tuple
from random import shuffle


def program_to_tokens(program: Program) -> Tuple[list[str], list[str]]:
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
    return (tokens, sorted(vocab))


def tokens_to_program(tokens: list[str]) -> Program:
    i = 0
    program = Program()
    while i+2 < len(tokens):
        type = Operation.Type[tokens[i].upper()]
        target = Operand(tokens[i+1])
        source = Operand(tokens[i+2])
        program.ops.append(Operation(type, target, source))
        i += 3
    return program


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


def split_program(program: Program) -> list[Program]:
    # Split at nops
    splitted = []
    next = Program()
    for op in program.ops:
        if op.type == Operation.Type.NOP:
            if len(next.ops) > 0:
                splitted.append(next)
                next = Program()
        else:
            next.ops.append(op)
    if len(next.ops) > 0:
        splitted.append(next)
    return splitted
