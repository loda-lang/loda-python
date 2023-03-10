# -*- coding: utf-8 -*-

from enum import Enum


class Operand:
    """Operand model and (de-)serialization.

    Operands consist of two members: `type` (enum) and `value` (int).
    The operand's type can be either `CONSTANT`, `DIRECT` memory access,
    or `INDIRECT` memory access. Constants are plain integers. Direct
    memory access is indicated using a single dollar sign, for example
    `$3`. Indirect memory access is indicated using a double dollar sign,
    for example `$$5`.

    **Example**
    >>> # Constructing operands using explicit types and values:
    >>> print(Operand(Operand.Type.CONSTANT, 1))
    1
    >>> print(Operand(Operand.Type.DIRECT, 3))
    $3
    >>> print(Operand(Operand.Type.INDIRECT, 5))
    $$5
    >>>
    >>> # Constructing operands from their string representations:
    >>> print(Operand("1"))
    1
    >>> print(Operand("$3"))
    $3
    >>> print(Operand("$$5"))
    $$5
    """

    class Type(Enum):
        """Enumeration for operand types. Supported types are
        `CONSTANT`, `DIRECT` and `INDIRECT`.
        """
        CONSTANT = 1
        """Used for constants."""
        DIRECT = 2
        """Direct memory access."""
        INDIRECT = 3
        """Indirect memory access."""

    type: Type
    """Type of this operand."""

    value: int
    """Value of this operand. If `type` is `DIRECT` or `INDIRECT`, the value must be non-negative."""

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
