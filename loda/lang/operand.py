# -*- coding: utf-8 -*-

from enum import Enum


class Operand:
    '''LODA operand representation'''

    class Type(Enum):
        CONSTANT = 1
        DIRECT = 2
        INDIRECT = 3

    def __init__(self, *args):
        if len(args) == 2:
            self.type = args[0]
            self.value = args[1]
        elif len(args) == 1:
            s = args[0].strip()
            if s.startswith("$$"):
                self.type = Operand.Type.INDIRECT
                self.value = int(s[2:])
            elif s.startswith("$"):
                self.type = Operand.Type.DIRECT
                self.value = int(s[1:])
            else:
                self.type = Operand.Type.CONSTANT
                self.value = int(s)
        else:
            self.type = Operand.Type.CONSTANT
            self.value = 0

    def __eq__(self, o: object) -> bool:
        return self.type == o.type and self.value == o.value

    def __str__(self) -> str:
        r = str(self.value)
        if self.type == Operand.Type.DIRECT:
            r = "$" + r
        elif self.type == Operand.Type.INDIRECT:
            r = "$$" + r
        return r
