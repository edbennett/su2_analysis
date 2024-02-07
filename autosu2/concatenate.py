from argparse import ArgumentParser
from pandas import read_csv, concat


def main():
    parser = ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = []

    for filename in args.filename:
        frames.append(read_csv(filename, sep="\s+"))

    full_data = concat(frames)

    full_data.to_csv(path_or_buf=args.output, sep="\t", na_rep="?", index=False)


if __name__ == "__main__":
    main()
