"""Miner class for finding integer sequence programs."""

from loda.oeis import PrefixIndex, Sequence
from loda.runtime import Evaluator, Interpreter


class Miner:

    def __init__(self, sequences: list, interpreter: Interpreter, generator):
        self.__index = PrefixIndex(sequences)
        self.__interpreter = interpreter
        self.__generator = generator

    def __call__(self):
        program = self.__generator()
        evaluator = Evaluator(program, self.__interpreter)
        match = self.__index.global_match()
        refine = True
        try:
            while refine:
                term = evaluator()
                # print(term)
                refine = self.__index.refine_match(match, term)
        except Exception as e:
            print("evaluation error: {}".format(e))
            return

        ids = self.__index.get_match_ids(match)
        for id in ids:
            print("Found match for {}".format(Sequence(id)))
            print(program)
