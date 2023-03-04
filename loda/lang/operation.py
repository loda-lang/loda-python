# -*- coding: utf-8 -*-

from enum import Enum
from .operand import Operand


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
        # Note: we ignore comments in equality checks.
        return self.type == o.type and self.target == o.target and \
            self.source == o.source

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
