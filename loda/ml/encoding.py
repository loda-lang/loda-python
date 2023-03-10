from loda.lang import Operand, Operation, Program
from loda.oeis import ProgramCache

import typing
import random


def program_to_tokens(program: Program) -> typing.Tuple[list[str], list[str]]:
    tokens = []
    vocab = set()
    for op in program.operations:
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
        program.operations.append(Operation(type, target, source))
        i += 3
    return program


def merge_programs(program_cache: ProgramCache, num_nop_seps: int = 1) -> Program:
    ids = program_cache.all_ids()
    random.shuffle(ids)
    merged = Program()
    for id in ids:
        program = program_cache.get(id)
        for _ in range(num_nop_seps):
            merged.operations.append(Operation())  # nop
        for op in program.operations:
            if op.type != Operation.Type.NOP:
                merged.operations.append(op)
    for _ in range(num_nop_seps):
        merged.operations.append(Operation())  # nop
    return merged


def split_program(program: Program) -> list[Program]:
    splitted = []
    next = Program()
    for op in program.operations:
        if op.type == Operation.Type.NOP:
            if len(next.operations) > 0:
                splitted.append(next)
                next = Program()
        else:
            next.operations.append(op)
    if len(next.operations) > 0:
        splitted.append(next)
    return splitted
