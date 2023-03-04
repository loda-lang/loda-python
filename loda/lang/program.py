# -*- coding: utf-8 -*-

from .operation import Operation


class Program:
    '''LODA program representation'''

    def __init__(self, pstr=None):
        self.ops = []
        if pstr:
            self.read(pstr)

    def read(self, pstr) -> None:
        self.ops = []
        for line in pstr.splitlines():
            self.ops.append(Operation(line))

    def __eq__(self, o: object) -> bool:
        return self.ops == o.ops

    def __str__(self) -> str:
        result = ""
        indent = 0
        for op in self.ops:
            if op.type == Operation.Type.LPE:
                indent = max(indent - 1, 0)
            result += "{}{}\n".format("  " * indent, op)
            if op.type == Operation.Type.LPB:
                indent += 1
        return result
