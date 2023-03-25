# -*- coding: utf-8 -*-

"""Program model and serialization."""

from .operation import Operation, Operand


class Program:
    """
    Programs are essentially a list of `Operation`s.

    >>> # Constructing programs from operations:
    >>> p1 = Program()
    >>> p1.ops.append(Operation("add $0,7"))
    >>> p1.ops.append(Operation("mul $0,$1"))
    >>> print(p1)
    >>> add $0,7
    >>> mul $0,$1
    >>>
    >>> # Constructing programs from strings representations:
    >>> p2 = Program("sub $0,5\\ndiv $0,$2")
    >>> print(p2)
    sub $0,5
    div $0,$2
    """

    operations: list
    """Operations of this program."""

    def __init__(self, program_str=None):
        self.operations = []
        if program_str:
            self.read(program_str)

    def read(self, program_str: str) -> None:
        """Read a program from a string representation. """
        self.operations = []
        for line in program_str.splitlines():
            self.operations.append(Operation(line))

    def __eq__(self, o: object) -> bool:
        return self.operations == o.operations

    def __str__(self) -> str:
        result = ""
        indent = 0
        for op in self.operations:
            if op.type == Operation.Type.LPE:
                indent = max(indent - 1, 0)
            result += "{}{}\n".format("  " * indent, op)
            if op.type == Operation.Type.LPB:
                indent += 1
        return result

    def validate(self):
        loop_depth = 0
        for op in self.operations:
            if op.type != Operation.Type.NOP and op.type != Operation.Type.LPE and op.target.type == Operand.Type.CONSTANT:
                raise ValueError("target cannot be a constant")
            if op.type == Operation.Type.LPB:
                loop_depth += 1
            elif op.type == Operation.Type.LPE:
                if loop_depth == 0:
                    raise ValueError("unexpected lpe")
                loop_depth -= 1
        if loop_depth != 0:
            raise ValueError("missing lpe")
