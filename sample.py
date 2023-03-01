import os.path

from loda.oeis import ProgramCache
from loda.runtime import Interpreter

# Sample program using the LODA Pathon module
if __name__ == "__main__":

    # Initialize LODA programs cache using *.asm files from tests folder
    program_dir = os.path.join('tests', 'programs', 'oeis')
    program_cache = ProgramCache(program_dir)

    # Load the LODA program for the prime numbers (A000040.asm)
    # See also the integer sequence entry at https://oeis.org/A000040
    program = program_cache.get(40)  # numeric version of A000040
    print(program)

    # Evaluate the program to an integer sequence
    interpreter = Interpreter(program_cache=program_cache)
    sequence, _ = interpreter.eval_to_seq(program, num_terms=20)
    print(sequence)
