#!/usr/bin/env python3

from argparse import ArgumentParser
from bioacoustic_detection.utils.annot import (
    clean_all_annotations_in_directory
)
def get_args():
    parser = ArgumentParser(
        description="Cleans annotation files in a directory, " \
        "extracting rejected annotations into separate files."
    )

    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        default=".",
        help="Input directory from which to read annotation files. " \
        "(Defaults to '.')"
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Output directory to write annotation clean/rejected files. " \
        "(Defaults to the input directory)"
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose printing."
    )

    return parser.parse_args()


def main(args):
    verbose = not args.quiet
    clean_all_annotations_in_directory(
        input_directory=args.in_dir,
        output_directory=args.out_dir,
        check_misnomers=verbose,
        verbose=verbose
    )
    if verbose:
        print("Finished cleaning annotations!")


if __name__ == "__main__":
    main(get_args())

