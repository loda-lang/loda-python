from loda.lang import Operand, Operation, Program
from loda.oeis import ProgramCache

import random


def program_to_tokens(program: Program):
    """
    Convert a program to tokens and vocabulary.

    Every operation is represented using three tokens:

    1. operation type
    2. target operand
    3. source operand

    Args:
        program: Program to be converted to tokens.
    Return:
        Pair of token list and vocabulary list.

    ## Example
    >>>
    >>> program = Program("mov $1,5\\ndiv $1,$3")
    >>> program_to_tokens(program)
    (['mov', '$1', '5', 'div', '$1', '$3'], ['$1', '$3', '5', 'div', 'mov'])
    """
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
    return tokens, sorted(vocab)


def tokens_to_operation(tokens: list, start: int):
    try:
        type = Operation.Type[tokens[start].upper()]
        target = Operand(tokens[start+1])
        source = Operand(tokens[start+2])
        return Operation(type, target, source)
    except Exception as e:
        return None


def tokens_to_program(tokens: list) -> Program:
    i = 0
    program = Program()
    while i+2 < len(tokens):
        operation = tokens_to_operation(tokens, i)
        program.operations.append(operation)
        i += 3
    return program


def append_nops(program: Program, num_nops: int):
    for _ in range(num_nops):
        program.operations.append(Operation())  # nop


def get_random_program_ids(program_cache: ProgramCache, num_programs: int = -1):
    # Get IDs of all existing programs. Shuffle them and reduce
    # the number of program IDs if requested.
    ids = program_cache.all_ids()
    random.shuffle(ids)
    if num_programs >= 0 and len(ids) > num_programs:
        ids = ids[0:num_programs]
    return ids


def merge_programs(program_cache: ProgramCache, program_ids: list,
                   num_ops_per_sample: int, num_nops_separator: int):

    # Merge all programs into one program. Invidual programs are
    # separated by (multiple) nops. The number nops equals the
    # number of operations per sample.
    merged = Program()
    num_loaded = 0
    for id in program_ids:
        program = program_cache.get(id)
        append_nops(merged, num_nops_separator)
        for op in program.operations:
            if op.type != Operation.Type.NOP:
                merged.operations.append(op)
        num_loaded += 1
        if num_loaded % 1000 == 0:
            program_cache.clear()
            # TODO: use logger
            print(num_loaded)
    program_cache.clear()

    # Calculate the sample size (includes input and label).
    # One operation is encoded using three tokens plus one
    # token because we split into input and label.
    sample_size = (3 * num_ops_per_sample) + 1

    # Extend the training dataset with nops such that samples are
    # get shifted by one when repeating the data set. We later want
    # to repeat the dataset such that we cover all possible start
    # positions. This code works because the sample size is 1 mod 3.
    shift = (3 * len(merged.operations)) % sample_size
    while shift != 1 and shift != (sample_size-1):
        append_nops(merged, 1)
        shift = (3 * len(merged.operations)) % sample_size
    num_samples = 3 * len(merged.operations)
    return merged, num_samples, sample_size


def split_sample(sample: list):
    # Split sample into (input,label) pair
    return sample[:-1], sample[1:]
