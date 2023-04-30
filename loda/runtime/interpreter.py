"""Interpret programs."""

from loda.lang import Operand, Operation, Program
from loda.oeis import ProgramCache
from .operations import exec_arithmetic


class Interpreter:

    def __init__(self,
                 program_cache: ProgramCache = None,
                 max_steps: int = -1,
                 max_memory: int = -1,
                 max_loop_stack_size: int = 100,
                 max_terms_cache_size: int = 10000):
        """
        The interpreter class is used to run LODA programs. Running programs operate on memory dictionaries
        mapping indices to integer values. The most common use case it to evaluate programs to integer sequences.

        Args:
            program_cache: Program cache (optional). When referencing programs using OEIS sequence IDs,
                the program cache must be supplied.
            max_steps: Maximum number of executions steps (-1 for no limit).
            max_loop_stack_size: Maximum stack size used for loops (-1 for no limit).
            max_terms_cache_size: Maximum cache size for sequence terms (-1 for no limit).

        ## Example
        >>> # Evaluate a program to an integer sequence:
        >>> fibonacci = Program(...)
        >>> interpreter.eval_to_seq(fibonacci, num_terms=10)
        ([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], 305)

        """
        self.__program_cache = program_cache
        self.__max_steps = max_steps
        self.__max_memory = max_memory
        self.__max_loop_stack_size = max_loop_stack_size
        self.__max_terms_cache_size = max_terms_cache_size
        self.__terms_cache = {}
        self.__running_programs = set()

    def clear_cache(self):
        """Clear the program and the terms cache. This is useful to free accumulated memory."""
        if self.__program_cache:
            self.__program_cache.clear()
        self.__terms_cache.clear()

    def eval_to_seq(self, program: Program, num_terms: int, use_steps=False):
        """
        Evaluate a program to an integer sequence.

        Args:
            program: The program to be evaluated.
            num_terms: The number of sequence terms to be computed.
            use_steps: Flag indicating whether to return the number of steps as sequence.
        Return:
            This function returns a pair. The first entry is the computed integer sequence
            as a list of ints. The second entry is the number of used computation steps.
        """
        seq = []
        mem = {}
        total_steps = 0
        for i in range(num_terms):
            mem.clear()
            mem[0] = i
            steps = self.run(program, mem)
            seq.append(steps if use_steps else mem[0])
            total_steps += steps
        return seq, total_steps

    def run(self, id_or_program, memory: dict):
        """Run a program."""
        if isinstance(id_or_program, Program):
            return self.__run(id_or_program, memory)
        elif isinstance(id_or_program, int):
            id = id_or_program
            if id in self.__running_programs:
                self.__raise("recursion detected")
            if self.__program_cache is None:
                self.__raise("program cache not set")
            program = self.__program_cache.get(id)
            if program is None:
                self.__raise("program not found: {}".format(id))
            self.__running_programs.add(id)
            steps = 0
            try:
                steps = self.__run(program, memory)
            finally:
                self.__running_programs.remove(id)
            return steps
        else:
            self.__raise(
                "expected ID or program: {}".format(id_or_program))

    def __run(self, p: Program, mem: dict) -> int:

        # remove nop operations
        ops = list(
            filter(lambda op: (op.type != Operation.Type.NOP), p.operations))

        # check for loops with fragments
        if any(op.type == Operation.Type.LPB and op.source != Operand(Operand.Type.CONSTANT, 1) for op in ops):
            self.__raise("fragments not supported")

        # check for empty program
        if len(ops) == 0:
            return 0

        # define stacks
        loop_stack = []
        counter_stack = []
        mem_stack = []

        steps = 0
        num_ops = len(ops)
        mem_tmp = mem.copy()
        mem_seq = {}

        # start program execution
        pc = 0
        while pc < num_ops:
            op = ops[pc]
            pc_next = pc + 1

            if op.type == Operation.Type.LPB:
                if len(loop_stack) >= self.__max_loop_stack_size and self.__max_loop_stack_size >= 0:
                    self.__raise(
                        "maximum stack size exceeded: {}".format(len(loop_stack)))
                loop_stack.append(pc)
                mem_stack.append(mem_tmp.copy())
                counter = self.__get(op.target, mem_tmp, False)
                counter_stack.append(counter)

            elif op.type == Operation.Type.LPE:
                lpb = ops[loop_stack[-1]]
                counter = self.__get(lpb.target, mem_tmp, False)
                if counter >= 0 and counter < counter_stack[-1]:
                    pc_next = loop_stack[-1] + 1  # jump back to begin
                    mem_stack[-1] = mem_tmp.copy()
                    counter_stack[-1] = counter
                else:
                    mem_tmp = mem_stack.pop()
                    loop_stack.pop()
                    counter_stack.pop()

            elif op.type == Operation.Type.SEQ:
                argument = self.__get(op.target, mem_tmp)
                seq_id = self.__get(op.source, mem_tmp)
                key = (seq_id, argument)
                if key in self.__terms_cache:
                    seq_result, seq_steps = self.__terms_cache[key]
                else:
                    mem_seq.clear()
                    mem_seq[0] = argument
                    seq_steps = self.run(seq_id, mem_seq)
                    seq_result = mem_seq.get(0, 0)
                    if self.__max_terms_cache_size < 0 or len(self.__terms_cache) <= self.__max_terms_cache_size:
                        self.__terms_cache[key] = (seq_result, seq_steps)
                self.__set(op.target, seq_result, mem_tmp, op)
                steps += seq_steps

            else:
                # arithmetic operation
                target = self.__get(op.target, mem_tmp)
                source = self.__get(op.source, mem_tmp)
                self.__set(op.target, exec_arithmetic(
                    op.type, target, source), mem_tmp, op)

            pc = pc_next

            # count execution steps
            steps += 1

            # check resource constraints
            if steps > self.__max_steps and self.__max_steps >= 0:
                self.__raise("exceeded maximum number of steps ({}); last operation: {}".format(
                    self.__max_steps, op))
            if len(mem_tmp) > self.__max_memory and self.__max_memory >= 0:
                self.__raise(
                    "exceeded maximum memory: {}; last operation: {} ".format(len(mem_tmp), op))

        # sanity check
        if len(loop_stack) + len(counter_stack) + len(mem_stack) > 0:
            self.__raise("execution error")

        # update main memory and return steps
        mem.clear()
        mem.update(mem_tmp)
        return steps

    def __get(self, op: Operand, mem, get_address=False):
        if op.type == Operand.Type.CONSTANT:
            if get_address:
                self.__raise("cannot get address of a constant")
            return op.value
        elif op.type == Operand.Type.DIRECT:
            return op.value if get_address else mem.get(op.value, 0)
        elif op.type == Operand.Type.INDIRECT:
            return mem.get(op.value, 0) if get_address else mem.get(mem.get(op.value, 0), 0)

    def __set(self, op: Operand, v, mem, last):
        index = op.value
        if op.type == Operand.Type.CONSTANT:
            self.__raise("cannot set value of a constant")
        elif op.type == Operand.Type.INDIRECT:
            index = mem.get(index, 0)
        if index > self.__max_memory and self.__max_memory >= 0:
            self.__raise(
                "maximum memory exceeded: {}; last operation: {}".format(index, last))
        if v == None:
            self.__raise(
                "overflow in {}; last operation: {}".format(op, last))
        mem[index] = v

    def __raise(self, msg: str) -> None:
        raise ValueError(msg)
