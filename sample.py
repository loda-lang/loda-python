import os.path

from loda.oeis import ProgramCache, sequence
from loda.runtime import Evaluator, Interpreter
from loda.mine import Miner
from loda.ml.keras.program_generation_rnn import load_model, train_model, Generator


class SampleLODA:

    def __init__(self):
        # Initialize LODA programs cache using *.asm files from tests folder
        programs_dir = os.path.join("tests", "programs", "oeis")
        # programs_dir = os.path.expanduser("~/loda/programs/oeis")
        self.program_cache = ProgramCache(programs_dir)
        self.interpreter = Interpreter(self.program_cache)

    def print_program(self):
        # Load the LODA program for the prime numbers (A000040.asm)
        # See also the integer sequence entry at https://oeis.org/A000040
        program = self.program_cache.get(40)  # numeric version of A000040
        print(program)

    def eval_program_to_seq(self):
        # Evaluate the program to an integer sequence
        program = self.program_cache.get(40)  # numeric version of A000040
        evaluator = Evaluator(program, self.interpreter)
        for _ in range(10):
            print(evaluator())

    def mine(self):
        model = train_model(self.program_cache, num_programs=1000)
        model.save("sample_model")

        # Load the model back from disk.
        loaded = load_model("sample_model")
        loaded.summary()

        # Use the trained model to generate programs.
        generator = Generator(loaded, num_lanes=10)

        for _ in range(10):
            print(generator())

        existing_ids = set(self.program_cache.all_ids())
        seqs = sequence.load(os.path.expanduser("~/loda/oeis"))
        seqs = list(filter(lambda s:
                           len(s.terms) >= 8 and s.id not in existing_ids, seqs))
        print("Loaded {} sequences".format(len(seqs)))
        miner = Miner(seqs, self.interpreter, generator)
        for i in range(500):
            miner()
            if i % 10 == 0:
                print(generator.get_stats_info_str())


if __name__ == "__main__":
    sample = SampleLODA()
    sample.print_program()
    sample.eval_program_to_seq()
    sample.mine()
