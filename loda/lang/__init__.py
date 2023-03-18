"""
Load and save programs.

This module contains the in-memory representation of LODA programs. You can use it to load and save programs,
and to inspect and manipulate their structure programmatically. You can load a program from an `*.asm` file
as follows:

>>> from loda.lang import Program
>>>
>>> with open("fibonacci.asm", "r") as file:
>>>     program = Program(file.read())
>>>     print(program)

To save it, just write the string representation of the program to another file.
To inspect and manipulate programs, see the `loda.lang.program.Program` class.
"""

from .operand import Operand
from .operation import Operation
from .program import Program
