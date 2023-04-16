"""Miner class for finding integer sequence programs."""

import os
import time

from loda.lang import Program
from loda.oeis import PrefixIndex, Sequence
from loda.runtime import Evaluator, Interpreter


class Miner:

    def __init__(self, sequences: list, interpreter: Interpreter, generator):
        self.__index = PrefixIndex(sequences)
        self.__interpreter = interpreter
        self.__generator = generator
        self.__basic_timeout = 60
        self.__extended_timeout = 600

    def __call__(self):
        program = self.__generator()
        evaluator = Evaluator(program, self.__interpreter)
        match = self.__index.global_match()
        refine = True
        start_time = time.time()
        try:
            while refine:
                term = evaluator()
                refine = self.__index.refine_match(match, term)
                if (time.time() - start_time) > self.__basic_timeout:
                    print("timeout")
                    return
        except Exception as e:
            print("evaluation error: {}".format(e))
            return
        ids = self.__index.get_match_ids(match)
        for id in ids:
            self.__check_match(program, id)

    def __check_match(self, program: Program, id: int):
        seq = self.__index.get(id)
        terms = seq.load_b_file(os.path.expanduser("~/loda/oeis"))
        if terms is None or len(terms) == len(seq.terms):
            print("Skipping check for {}".format(seq))
            return
        print("Checking match for {}".format(seq))
        start_time = time.time()
        if len(terms) > 1000:
            terms = terms[0:1000]
        evaluator = Evaluator(program, self.__interpreter)
        correct = True
        for t in terms:
            n = evaluator()
            if n != t:
                correct = False
                break
            if (time.time() - start_time) > self.__extended_timeout:
                print("timeout")
                return
        if correct:
            print("Found match for {}".format(seq))
            p = "~/loda/programs/local/{}.asm".format(seq.id_str())
            with open(os.path.expanduser(p), "w") as asm_file:
                asm_file.write("; {}\n".format(seq))
                asm_file.write(str(program))
