from .io_utils import read_annotations, save_annotations
import glob
import os.path as path
import numpy as np


class AnnotationFormat:
    """
    Class containing useful data for accessing and manipulating annotations.
    """
    LEFT_COL = "Begin Time (s)"
    RIGHT_COL = "End Time (s)"
    TOP_COL = "High Freq (Hz)"
    BOT_COL = "Low Freq (Hz)"
    CLASS_COL = "Species"
    CLASS_CONF_COL = "Species Confidence"
    CALL_UNCERTAINTY_COL = "Call Uncertainty"

    # Useful glob patterns
    PATTERN = "*.*-*.txt"
    BAD_PATTERNS = ["*.*_*.txt"]

_format = AnnotationFormat()


def get_all_classes(annotation_paths, verbose=False):
    """
    Returns a list of all classes seen in the annotation files.

    Parameters
    annotation_paths : list of str
        paths to the .txt annotation files (eg: ['/foo/bar/annots.txt'])
    verbose : bool, optional (default: False)
        flag to control whether debug information is printed
    
    Returns
    classes : list of str
        List containing all unique classes
    """

    classes = set()
    for annot_fname in annotation_paths:
        classes.update(list(read_annotations(annot_fname)[_format.CLASS_COL].unique()))
    classes = sorted([s for s in list(classes)])
    if verbose:
        print("Classes: ", classes)
    return classes


def get_area(annotation):
    """
    Calculates the area of a single annotation box.

    Parameters
    annotation : pandas Series
        a single annotation
    
    Returns
    area : float
        Area of the bounding box (Hz*Seconds)
    """

    return ((annotation[_format.RIGHT_COL] - annotation[_format.LEFT_COL])
            * (annotation[_format.TOP_COL] - annotation[_format.BOT_COL]))


def get_all_annotations_in_directory(directory, check_misnomers=True):
    """
    Uses glob to construct a list of paths to each file in the provided
    directory which matches the correct formatting of an annotation file name.

    Parameters
    directory : str
        path to the directory of interest
    check_misnomers : bool, optional (default: True)
        flag to control whether to warn about potential filename mistakes
    
    Returns
    good_results : List of str
        Paths found in the given directory which match the filename pattern
    """

    good_results = glob.glob(path.join(directory, _format.PATTERN))

    if check_misnomers:
        # Check if there are any incorrectly named files that may be overlooked
        bad_results = []
        for bad_pattern in _format.BAD_PATTERNS:
            bad_results.extend(glob.glob(path.join(directory, bad_pattern)))
        if len(bad_results) > 0:
            raise RuntimeWarning(
                "({}) Some files in {} may be incorrectly named: " \
                "[\n  {}\n]".format(
                    "get_all_annotations_in_directory",
                    directory,
                    ",\n  ".join(bad_results)
                )
            )

    return good_results


def levenshteinDistanceDP(token1, token2):
    """
    Efficiently calculates the Levenshtein distance (edit distance) between two
    strings. Useful for determining if a column name has been misspelled.

    The cost of insertions, deletions, and substitutions are all set to 1.

    Parameters
    token1 : str
        first token
    token2 : str
        second token
    
    Returns
    distance : int
        the number of single-character edits required to turn token1 into token2
    """

    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a, b, c = 0, 0, 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def clean_annotations(annotations, verbose=False):
    """
    Cleans a single DataFrame of annotations by identifying invalid annotations
    and separating them from the valid annotations.

    Parameters
    annotations : DataFrame
        a set of annotations from a single recording
    verbose : bool, optional (default: False)
        flag to control whether debug information is printed
    
    Returns
    valid_annotations : DataFrame
        the annotations that passed every filter
    invalid_annotations : DataFrame
        the annotations that failed at least one filter
    """

    raise NotImplementedError()


def clean_all_annotations_in_directory(
        input_directory,
        output_directory=None,
        check_misnomers=True,
        verbose=True):
    """
    Cleans every annotation file in a directory and saves the invalid
    annotations to their own file so that they can be checked and fixed.

    Invalid annotations will be saved in a file with "_rejected" placed
    immediately before the file extension.
    For example: "foo.bar-AB.txt" --> "foo.bar-AB_rejected.txt

    Parameters
    input_directory : str
        path to the directory containing annotation files to clean
    output_directory : str (defaults to match input_directory)
        path to output valid and invalid annotation files
    check_misnomers : bool, optional (default: True)
        flag to control whether to warn about potential filename mistakes
    verbose : bool, optional (default: True)
        flag to control whether debug information is printed
    """

    if output_directory is None:
        output_directory = input_directory

    annotation_paths = get_all_annotations_in_directory(
        input_directory,
        check_misnomers=check_misnomers
    )

    for annot_path in annotation_paths:
        annotations = read_annotations(annot_path, verbose=verbose)
        valid_annotations, invalid_annotations = \
            clean_annotations(annotations, verbose=verbose)
        valid_annots_path = path.join(
            output_directory,
            path.basename(annot_path)
        )
        save_annotations(
            valid_annotations,
            valid_annots_path,
            verbose=verbose
        )
        invalid_annots_path = path.join(
            output_directory,
            "{}_rejected{}".format(*path.splitext(path.basename(annot_path)))
        )
        save_annotations(
            invalid_annotations,
            invalid_annots_path,
            verbose=verbose
        )