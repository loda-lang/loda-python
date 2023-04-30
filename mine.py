import datetime
import os.path

from loda.lang import Operation, Program
from loda.oeis import ProgramCache, Sequence
from loda.runtime import Interpreter
from loda.mine import Miner
from loda.ml.keras.program_generation_rnn import load_model, Generator


def mine(model_path: str):

    model = load_model(model_path)
    model.summary()
    initial_program = Program()
    initial_program.operations.append(Operation("mov $1,1"))
    generator = Generator(model, initial_program=initial_program, num_lanes=10)
    programs_dir = os.path.expanduser("~/loda/programs/oeis")
    program_cache = ProgramCache(programs_dir)
    ignore_ids = set()
    ignore_ids.update(program_cache.all_ids())
    ignore_ids.update(Sequence.load_id_list(
        os.path.expanduser("~/loda/programs/oeis/deny.txt")))
    seqs = Sequence.load_oeis(os.path.expanduser("~/loda/oeis"))
    seqs = list(filter(lambda s: len(s.terms) >=
                8 and s.id not in ignore_ids, seqs))
    print("Loaded {} sequences".format(len(seqs)))
    interpreter = Interpreter(program_cache)
    miner = Miner(seqs, interpreter, generator)
    i = 0
    while True:
        miner()
        i += 1
        if i % 10 == 0:
            ct = datetime.datetime.now()
            print(ct, generator.get_stats_info_str())


if __name__ == "__main__":
    mine(os.path.expanduser("~/scripts/model-075"))
