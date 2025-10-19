# -*- coding: utf-8 -*-

"""Test memory operations: clr, fil, rol, ror"""

from unittest import TestCase
from loda.lang import Operation, Operand, Program
from loda.runtime import Interpreter


class MemoryOperationsTests(TestCase):

    def setUp(self):
        self.interpreter = Interpreter(max_memory=10000)

    def test_clr_positive_length(self):
        """Test clear operation with positive length"""
        program = Program("""
mov $1,5
mov $2,10
mov $3,15
clr $1,2
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 0)
        self.assertEqual(mem.get(2, 0), 0)
        self.assertEqual(mem.get(3, 0), 15)

    def test_clr_negative_length(self):
        """Test clear operation with negative length"""
        program = Program("""
mov $1,5
mov $2,10
mov $3,15
clr $3,-2
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 5)
        self.assertEqual(mem.get(2, 0), 0)
        self.assertEqual(mem.get(3, 0), 0)

    def test_fil_positive_length(self):
        """Test fill operation with positive length"""
        program = Program("""
mov $1,7
mov $2,10
mov $3,15
fil $1,3
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 7)
        self.assertEqual(mem.get(2, 0), 7)
        self.assertEqual(mem.get(3, 0), 7)

    def test_fil_negative_length(self):
        """Test fill operation with negative length"""
        program = Program("""
mov $1,5
mov $2,10
mov $3,20
fil $3,-2
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 5)
        self.assertEqual(mem.get(2, 0), 20)
        self.assertEqual(mem.get(3, 0), 20)

    def test_rol_positive_length(self):
        """Test rotate left operation with positive length"""
        program = Program("""
mov $1,10
mov $2,20
mov $3,30
rol $1,3
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 20)
        self.assertEqual(mem.get(2, 0), 30)
        self.assertEqual(mem.get(3, 0), 10)

    def test_rol_negative_length(self):
        """Test rotate left operation with negative length"""
        program = Program("""
mov $1,10
mov $2,20
mov $3,30
rol $3,-2
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 10)
        self.assertEqual(mem.get(2, 0), 30)
        self.assertEqual(mem.get(3, 0), 20)

    def test_ror_positive_length(self):
        """Test rotate right operation with positive length"""
        program = Program("""
mov $1,10
mov $2,20
mov $3,30
ror $1,3
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 30)
        self.assertEqual(mem.get(2, 0), 10)
        self.assertEqual(mem.get(3, 0), 20)

    def test_ror_negative_length(self):
        """Test rotate right operation with negative length"""
        program = Program("""
mov $1,10
mov $2,20
mov $3,30
ror $3,-2
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 10)
        self.assertEqual(mem.get(2, 0), 30)
        self.assertEqual(mem.get(3, 0), 20)

    def test_memory_operation_with_indirect_addressing(self):
        """Test memory operations with indirect addressing"""
        program = Program("""
mov $1,2
mov $2,100
mov $3,200
fil $$1,2
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(2, 0), 100)
        self.assertEqual(mem.get(3, 0), 100)

    def test_max_memory_limit_fil(self):
        """Test that memory range limit is enforced for fil operation"""
        program = Program("""
mov $1,100
fil $1,20000
""")
        mem = {0: 0}
        with self.assertRaises(ValueError) as context:
            self.interpreter.run(program, mem)
        self.assertIn("maximum memory exceeded", str(context.exception).lower())

    def test_max_memory_limit_rol(self):
        """Test that memory range limit is enforced for rol operation"""
        program = Program("""
mov $1,100
rol $1,20000
""")
        mem = {0: 0}
        with self.assertRaises(ValueError) as context:
            self.interpreter.run(program, mem)
        self.assertIn("maximum memory exceeded", str(context.exception).lower())

    def test_max_memory_limit_ror(self):
        """Test that memory range limit is enforced for ror operation"""
        program = Program("""
mov $1,100
ror $1,20000
""")
        mem = {0: 0}
        with self.assertRaises(ValueError) as context:
            self.interpreter.run(program, mem)
        self.assertIn("maximum memory exceeded", str(context.exception).lower())

    def test_rol_zero_length(self):
        """Test rotate left with zero length (should do nothing)"""
        program = Program("""
mov $1,10
mov $2,20
rol $1,0
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 10)
        self.assertEqual(mem.get(2, 0), 20)

    def test_ror_zero_length(self):
        """Test rotate right with zero length (should do nothing)"""
        program = Program("""
mov $1,10
mov $2,20
ror $1,0
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(1, 0), 10)
        self.assertEqual(mem.get(2, 0), 20)

    def test_clr_with_sparse_memory(self):
        """Test clear operation with sparse memory (non-consecutive addresses)"""
        program = Program("""
mov $10,100
mov $20,200
mov $30,300
clr $10,25
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(10, 0), 0)
        self.assertEqual(mem.get(20, 0), 0)
        self.assertEqual(mem.get(30, 0), 0)

    def test_fil_with_constant_source(self):
        """Test fill operation with constant as length"""
        program = Program("""
mov $5,42
fil $5,3
""")
        mem = {0: 0}
        self.interpreter.run(program, mem)
        self.assertEqual(mem.get(5, 0), 42)
        self.assertEqual(mem.get(6, 0), 42)
        self.assertEqual(mem.get(7, 0), 42)
