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


class Operation:
    '''LODA operation representation'''

    class Type(Enum):
        NOP = 1
        MOV = 2
        ADD = 3
        SUB = 4
        TRN = 5
        MUL = 6
        DIV = 7
        DIF = 8
        MOD = 9
        POW = 10
        GCD = 11
        BIN = 12
        CMP = 13
        MIN = 14
        MAX = 15
        LPB = 16
        LPE = 17
        CLR = 18
        SEQ = 19
        DBG = 20

    def __init__(self, *args):
        if len(args) == 0:
            self.type = Operation.Type.NOP
            self.target = Operand()
            self.source = Operand()
            self.comment = None
        elif len(args) == 1:
            # parse operation
            s = args[0].strip()
            # extract comment
            i = s.find(";")
            if i >= 0:
                self.comment = s[i+1:].strip()
                s = s[0:i].strip()
            else:
                self.comment = None
            if s:
                # parse operation type
                t = s.split()[0]
                self.type = Operation.Type[t.upper()]
                s = s[len(t):].strip()
                # parse arguments
                a = s.split(",")
                self.target = Operand()
                self.source = Operand()
                if len(a) > 0 and s:
                    self.target = Operand(a[0])
                if len(a) > 1:
                    self.source = Operand(a[1])
                elif self.type == Operation.Type.LPB:
                    self.source = Operand(Operand.Type.CONSTANT, 1)
            else:
                self.type = Operation.Type.NOP
                self.target = Operand()
                self.source = Operand()
        elif len(args) == 3:
            self.type = args[0]
            self.target = args[1]
            self.source = args[2]
            self.comment = None

    def __eq__(self, o: object) -> bool:
        return self.type == o.type and self.target == o.target and \
            self.source == o.source and self.comment == o.comment

    def __str__(self) -> str:
        type_str = self.type.name.lower()
        if self.type == Operation.Type.NOP:
            r = ""
        elif self.type == Operation.Type.LPB and self.source == Operand(Operand.Type.CONSTANT, 1):
            r = "{} {}".format(type_str, self.target)
        elif self.type == Operation.Type.LPE:
            r = type_str
        else:
            r = "{} {},{}".format(type_str, self.target, self.source)
        if self.comment:
            if r:
                r += " "
            r += "; {}".format(self.comment)
        return r


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
