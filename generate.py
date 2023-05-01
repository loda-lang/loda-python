import argparse
import datetime
import os.path
import sys

from loda.lang import Program
from loda.ml.keras.program_generation_rnn import load_model, Generator


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def generate_programs(generator, num_programs: int, use_line_format: bool, write_fn, verbose=0):
    for i in range(num_programs):
        p = generator()
        if use_line_format:
            p = "; ".join([str(op) for op in p.operations])
        write_fn("{}\n".format(p))
        if verbose > 0 and i % 10 == 0:
            ct = datetime.datetime.now()
            eprint(ct, generator.get_stats_info_str())


def generate(model_path: str, output_path=None, num_programs=100, format="asm", verbose=0):
    model = load_model(model_path)
    if verbose > 0:
        model.summary(print_fn=eprint)
    initial_program = Program()
    # initial_program.operations.append(Operation("mov $1,1"))
    num_lanes = 10
    if num_programs >= 1000:
        num_lanes = 100
    elif num_programs >= 10000:
        num_lanes = 1000
    generator = Generator(
        model, initial_program=initial_program, num_lanes=100)
    use_line_format = (format == "line")
    if output_path:
        with open(output_path, "w") as file:
            generate_programs(generator, num_programs,
                              use_line_format, file.write, verbose)
    else:
        generate_programs(generator, num_programs,
                          use_line_format, sys.stdout.write, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument(
        "-f", "--format", type=str, choices=["asm", "line"], help="output format of the generated programs")
    parser.add_argument(
        "-o", "--output", type=str, help="output file for writing the programs to")
    parser.add_argument(
        "-n", type=int, help="number of programs to generate", default=100)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    generate(model_path=args.model, output_path=args.output,
             num_programs=args.n, format=args.format, verbose=args.verbose)
