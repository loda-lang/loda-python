from loda.lang import Operation, Program

import tensorflow as tf


def encode_program(program: Program) -> tf.RaggedTensor:
    tokens = []
    for op in program.ops:
        tokens.append(op.type.name.lower())
        tokens.append(str(op.target))
        tokens.append(str(op.source))
    return tf.ragged.constant([tokens])
