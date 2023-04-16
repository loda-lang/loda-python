import os.path

from loda.oeis import ProgramCache
from loda.ml.keras.program_generation_rnn import train_model


def train(programs_percentage: int):
    programs_dir = os.path.expanduser("~/loda/programs/oeis")
    program_cache = ProgramCache(programs_dir)
    num_train_programs = -1
    if programs_percentage < 100:
        num_total_programs = len(program_cache.all_ids())
        num_train_programs = (programs_percentage * num_total_programs) // 100
    print("Training using {} programs".format(num_train_programs))
    model = train_model(program_cache, num_programs=num_train_programs)
    model.save("model-{:03}".format(programs_percentage))


if __name__ == "__main__":
    train(1)
    train(25)
    train(50)
    train(100)
