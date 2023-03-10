# -*- coding: utf-8 -*-

from loda.lang import Operand, Operation, Program
from .operations import exec_arithmetic


class Interpreter:

    def __init__(self, program_cache=None, max_cycles=-1, max_memory=-1, max_terms_cache_size=10000, max_loop_stack_size=100):
        self.max_cycles = max_cycles
        self.max_memory = max_memory
        self.__program_cache = program_cache
        self.__terms_cache = {}
        self.__max_terms_cache_size = max_terms_cache_size
        self.__max_loop_stack_size = max_loop_stack_size
        self.__running_programs = set()

    def clear_cache(self) -> None:
        if self.__program_cache:
            self.__program_cache.clear()
        self.__terms_cache.clear()

    def eval_to_seq(self, p: Program, num_terms: int, use_cycles=False):
        seq = []
        mem = {}
        total_cycles = 0
        for i in range(num_terms):
            mem.clear()
            mem[0] = i
            cycles = self.run(p, mem)
            seq.append(cycles if use_cycles else mem[0])
            total_cycles += cycles
        return seq, total_cycles

    def run(self, id_or_program, mem):
        if isinstance(id_or_program, Program):
            return self.__run(id_or_program, mem)
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
            cycles = 0
            try:
                cycles = self.__run(program, mem)
            finally:
                self.__running_programs.remove(id)
            return cycles
        else:
            self.__raise(
                "expected ID or program: {}".format(id_or_program))

    def __run(self, p: Program, mem: dict) -> int:

        # remove nop operations
        ops = list(
            filter(lambda op: (op.type != Operation.Type.NOP), p.operations))

        # check for loops with fragments
        if any(op.type == Operation.Type.LPB and op.source != Operand(Operand.Type.CONSTANT, 1) for op in ops):
            self.__raise("unsupported operation: {}".format(op))

        # check for empty program
        if len(ops) == 0:
            return 0

        # define stacks
        loop_stack = []
        counter_stack = []
        mem_stack = []

        cycles = 0
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
                    seq_result, seq_cycles = self.__terms_cache[key]
                else:
                    mem_seq.clear()
                    mem_seq[0] = argument
                    seq_cycles = self.run(seq_id, mem_seq)
                    seq_result = mem_seq.get(0, 0)
                    if self.__max_terms_cache_size < 0 or len(self.__terms_cache) <= self.__max_terms_cache_size:
                        self.__terms_cache[key] = (seq_result, seq_cycles)
                self.__set(op.target, seq_result, mem_tmp, op)
                cycles += seq_cycles

            else:
                # arithmetic operation
                target = self.__get(op.target, mem_tmp)
                source = self.__get(op.source, mem_tmp)
                self.__set(op.target, exec_arithmetic(
                    op.type, target, source), mem_tmp, op)

            pc = pc_next

            # count execution steps
            cycles += 1

            # check resource constraints
            if cycles > self.max_cycles and self.max_cycles >= 0:
                self.__raise("exceeded maximum number of cycles ({}); last operation: {}".format(
                    self.max_cycles, op))
            if len(mem_tmp) > self.max_memory and self.max_memory >= 0:
                self.__raise(
                    "exceeded maximum memory: {}; last operation: {} ".format(len(mem_tmp), op))

        # sanity check
        if len(loop_stack) + len(counter_stack) + len(mem_stack) > 0:
            self.__raise("execution error")

        # update main memory and return cycles
        mem.clear()
        mem.update(mem_tmp)
        return cycles

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
        if index > self.max_memory and self.max_memory >= 0:
            self.__raise(
                "maximum memory exceeded: {}; last operation: {}".format(index, last))
        if v == None:
            self.__raise(
                "overflow in {}; last operation: {}".format(op, last))
        mem[index] = v

    def __raise(msg: str) -> None:
        raise ValueError(msg)
