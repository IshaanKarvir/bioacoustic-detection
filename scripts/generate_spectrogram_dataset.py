#!/usr/bin/env python3

from argparse import ArgumentParser
from bioacoustic_detection.spectrogram_dataset_generator import (
    generate_dataset
)

def get_args():
    parser = ArgumentParser(
        description="Generates a spectrogram dataset from the provided " \
        "recordings, annotations, and parameters. "
    )

    # FILE PATHS AND NAMES

    parser.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Input .json file which lists the dataset's training and evaluation splits."
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="Output directory to write the dataset."
    )

    parser.add_argument(
        "-l",
        "--label_map_name",
        type=str,
        default="label_map.pbtxt",
        help="Name of output label map file."
    )

    parser.add_argument(
        "-m",
        "--metadata_name",
        type=str,
        default="dataset_metadata.txt",
        help="Name of output metadata file."
    )

    # SPECTROGRAM CONSTANTS

    parser.add_argument(
        "--window_size_sec",
        type=float,
        default=3/20,
        help="Window size (n_fft) in seconds."
    )

    parser.add_argument(
        "--hop_len_sec",
        type=float,
        default=15/300,
        help="Hop length in seconds."
    )

    parser.add_argument(
        "--n_mels",
        type=int,
        default=300,
        help="Number of frequency bands (y dimension of spectrogram)."
    )

    parser.add_argument(
        "--freq_max",
        type=float,
        default=1600,
        help="Maximum frequency included in spectrograms (Hz)."
    )

    # CHUNK CONSTANTS

    parser.add_argument(
        "--train_chunk_len_sec",
        type=float,
        default=45.0,
        help="Length of one training chunk in seconds."
    )

    parser.add_argument(
        "--eval_chunk_len_sec",
        type=float,
        default=15.0,
        help="Length of one training chunk in seconds."
    )

    parser.add_argument(
        "--min_box_percent",
        type=float,
        default=0.3,
        help="Minimum percent visibility of a call to keep annotation."
    )

    # DATASET SETTINGS

    parser.add_argument(
        "--n_train_shards",
        type=int,
        default=3,
        help="Number of .tfrecord shards to break the train split into."
    )

    parser.add_argument(
        "--n_eval_shards",
        type=int,
        default=5,
        help="Number of .tfrecord shards to break the eval split into."
    )

    parser.add_argument(
        '-c',
        '--classes',
        nargs='+',
        default=["hb"],
        help="List of classes to include in the dataset. (Defaults to only 'hb')"
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
    generate_dataset(
        splits=args.splits,
        output_directory=args.out_dir,
        label_map_name=args.label_map_name,
        metadata_name=args.metadata_name,
        window_size_sec=args.window_size_sec,
        hop_len_sec=args.hop_len_sec,
        n_mels=args.n_mels,
        freq_max=args.freq_max,
        train_chunk_len_sec=args.train_chunk_len_sec,
        eval_chunk_len_sec=args.eval_chunk_len_sec,
        min_box_percent=args.min_box_percent,
        n_train_shards=args.n_train_shards,
        n_eval_shards=args.n_eval_shards,
        allowed_classes=args.classes,
        verbose=verbose
    )
    if verbose:
        print("Finished generating spectrogram dataset!")


if __name__ == "__main__":
    main(get_args())

