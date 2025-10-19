# -*- coding: utf-8 -*-

"""Operation model and serialization."""

from enum import Enum
from .operand import Operand


class Operation:
    """
    Operations have the following structure: `<type> <target>,<source> ; <comment>`
    where the operation type is an enum consisting of three-letter operation names, and
    target and source are `loda.lang.operand.Operand`s. Depending on their type, 
    operations may have no target/source operands. Note that the _first_ operand
    is the target, and the _second_ is the source, which is analogous to the 
    [Intel assembly syntax](https://en.wikipedia.org/wiki/X86_assembly_language).
    Comments are optional strings separated from the rest of the operation using a colon.

    The target operand can be a direct or indirect memory access, but never a constant.
    There are no type restrictions for the source operand. In terms of execution semantics,
    operations may update the content of the target memory cell, but the source is always read only.
    The precise execution semantics of all operations is defined in the
    [Language Specification](https://loda-lang.github.io/spec)
    and implemented in `loda.runtime` package.

    **Example**
    >>> # Example operands:
    >>> target = Operand("$1")
    >>> source = Operand("5")
    >>>
    >>> # Constructing operations using explicit values:
    >>> print(Operation(Operation.Type.MOV, target, source))
    mov $1,5
    >>>
    >>> print(Operation(Operation.Type.ADD, target, source, "some comment"))
    add $1,5 ; some comment
    >>>
    >>> # Constructing operations from their string representations:
    >>> print(Operation("sub $2,7"))
    sub $2,7
    >>> print(Operation("div $3,$5 ; some comment"))
    div $3,$5 ; some comment
    """

    class Type(Enum):
        """Operation type. Type names are written as three-letter, lower-case words."""
        NOP = 1     # no operation
        MOV = 2     # assignment
        ADD = 3     # addition
        SUB = 4     # subtraction
        TRN = 5     # truncated subtraction
        MUL = 6     # multiplication
        DIV = 7     # division
        DIF = 8     # conditional division
        DIR = 9     # repeated division
        MOD = 10    # modulo
        POW = 11    # power
        GCD = 12    # greatest common divisor
        LEX = 13    # largest exponent
        BIN = 14    # binomial coefficient
        FAC = 15    # falling and rising factorial
        LOG = 16    # discrete logarithm
        NRT = 17    # discrete n-th root
        DGS = 18    # digit sum
        DGR = 19    # digital root
        EQU = 20    # equality
        NEQ = 21    # inequality
        LEQ = 22    # less or equal
        GEQ = 23    # greater or equal
        MIN = 24    # minimum
        MAX = 25    # maximum
        BAN = 26    # bitwise and
        BOR = 27    # bitwise or
        BXO = 28    # bitwise xor
        LPB = 29    # loop begin
        LPE = 30    # loop end
        CLR = 31    # clear
        SEQ = 32    # sequence
        DBG = 33    # debug
        FIL = 34    # fill
        ROL = 35    # rotate left
        ROR = 36    # rotate right

    type: Type
    """Type of this operation."""

    target: Operand
    """Target operand."""

    source: Operand
    """Source operand."""

    comment: str
    """Optional comment."""

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
        elif len(args) == 4:
            self.type = args[0]
            self.target = args[1]
            self.source = args[2]
            self.comment = args[3]

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
