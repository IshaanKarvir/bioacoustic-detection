import numpy as np
import librosa
import json
import os
from os import path

from PIL import Image

# For producing label map for TF2 Obj Detection
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format

# For saving TFRecords
import tensorflow as tf
from object_detection.utils import dataset_util

# For sharding TFRecords
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

from bioacoustic_detection.utils.annot import (
    _format as annot_format,
    get_all_classes,
    get_area
)

from bioacoustic_detection.utils.wav import (
    PCEN
)

from bioacoustic_detection.utils.io import (
    read_wavfile,
    read_annotations
)


# Generates the necessary prototext file for the class mapping.
# Classes are assigned to the integer 1 greater than their index.
# The resulting file is saved to output_path.
def create_label_map(classes, output_path):
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    for i, cls in enumerate(classes):
        new_item = label_map.item.add() # StringIntLabelMapItem
        new_item.name = cls          # String name.
        new_item.id = 1+i            # Integer id starting from 1
        new_item.display_name = cls  # Human readable text label
    with open(output_path, "w") as f:
        f.write(text_format.MessageToString(label_map))


def get_splits(split_fname):
    # read JSON of splits and return lists of (wav, annot) pairs
    with open(split_fname) as split_file:
        splits = json.load(split_file)
    assert ("train" in splits and "eval" in splits), \
        "splits json must contain both train and eval splits"
    return splits["train"], splits["eval"]


# Returns the min and max values observed in all wav files
def get_minmax_bounds(wav_filenames,
                      window_size_sec,
                      hop_len_sec,
                      n_mels,
                      freq_max,
                      chunk_size):
    min_val, max_val = None, None
    for wfname in wav_filenames:
        sr, data = read_wavfile(wfname, normalize=True)
        n_fft = int(window_size_sec * sr)
        hop_len = int(hop_len_sec * sr)
        chunk_size = int(chunk_size * sr)
        step = chunk_size - (hop_len * (n_mels-2) + n_fft)
        M_init = None
        for start_i in range(0, len(data), step):
            end_i = min(len(data),start_i+chunk_size)
            mel_spec = librosa.feature.melspectrogram(y=data[start_i:end_i],
                                                      sr=sr,
                                                      n_fft=n_fft,
                                                      hop_length=hop_len,
                                                      n_mels=n_mels,
                                                      fmax=freq_max,
                                                      center=False)
            mel_spec, M_init = PCEN(mel_spec, step // hop_len, init_M=M_init)
            temp_min = mel_spec.min()
            temp_max = mel_spec.max()
            if min_val is None or temp_min < min_val:
                min_val = temp_min
            if max_val is None or temp_max > max_val:
                max_val = temp_max
    return min_val, max_val


# Creates the tf.train.Example from the example_dict saved for each spectrogram chunk
# The image bytes are read in from disk at this point.
def create_tf_example(example_dict):
    with open(example_dict["filepath"], "rb") as f:
        encoded_image_data = f.read()
    filename = path.basename(example_dict["filepath"]).encode()
    classes_text = [s.encode() for s in example_dict["classes_text"]]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
              'image/height': dataset_util.int64_feature(example_dict["height"]),
              'image/width': dataset_util.int64_feature(example_dict["width"]),
              'image/filename': dataset_util.bytes_feature(filename),
              'image/source_id': dataset_util.bytes_feature(filename),
              'image/encoded': dataset_util.bytes_feature(encoded_image_data),
              'image/format': dataset_util.bytes_feature(b'png'),
              'image/object/bbox/xmin': dataset_util.float_list_feature(example_dict["xmins"]),
              'image/object/bbox/xmax': dataset_util.float_list_feature(example_dict["xmaxs"]),
              'image/object/bbox/ymin': dataset_util.float_list_feature(example_dict["ymins"]),
              'image/object/bbox/ymax': dataset_util.float_list_feature(example_dict["ymaxs"]),
              'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
              'image/object/class/label': dataset_util.int64_list_feature(example_dict["classes"]),
          }))
    return tf_example


def process_file(wav_filename,
                 annot_filename,
                 min_bound,
                 max_bound,
                 chunk_size,
                 window_size_sec,
                 hop_len_sec,
                 n_mels,
                 freq_max,
                 min_box_percent,
                 output_directory,
                 classes,
                 rev_class_map,
                 chunk_layout="dense",
                 drop_last_chunk=False,
                 verbose=False):
    sr, data = read_wavfile(wav_filename, normalize=True, verbose=verbose)
    annotations = read_annotations(annot_filename, verbose=verbose)
    file_id = path.basename(wav_filename)[:22]
    
    n_fft = int(window_size_sec * sr)
    hop_len = int(hop_len_sec * sr)
    chunk_size = int(chunk_size * sr)

    # Get annotation format:
    left_col = annot_format.LEFT_COL
    right_col = annot_format.RIGHT_COL
    top_col = annot_format.TOP_COL
    bot_col = annot_format.BOT_COL
    class_col = annot_format.CLASS_COL

    
    if chunk_layout == "dense":
        step = chunk_size - (hop_len * (n_mels-2) + n_fft)
    elif chunk_layout == "sparse":
        step = chunk_size // 2
    
    # Start Indices of each chunk
    start_vals = [s for s in range(0, len(data), step)]
    
    # If last cut point creates a tiny chunk, remove it
    if len(data) - start_vals[-1] < int(chunk_size / 2):
        start_vals = start_vals[:-1]
        
    def extract_chunk(start_i, end_i, spec_name, M_init=None):
        mel_spec = librosa.feature.melspectrogram(y=data[start_i:end_i],
                                                  sr=sr,
                                                  n_fft=n_fft,
                                                  hop_length=hop_len,
                                                  n_mels=n_mels,
                                                  fmax=freq_max,
                                                  center=False)
        mel_spec, next_M_init = PCEN(mel_spec, step // hop_len, init_M=M_init)
        mel_spec = np.clip(
            (mel_spec - min_bound) / (max_bound - min_bound) * 255,
            a_min=0,
            a_max=255
        ).astype(np.uint8)
        spec_height, spec_width = mel_spec.shape
        
        # Get annotations inside chunk
        start_s, end_s = start_i/sr, end_i/sr
        freq_axis_low = librosa.hz_to_mel(0.0)
        freq_axis_high = librosa.hz_to_mel(freq_max)
        chunk_annotations = \
            annotations.loc[~((annotations[left_col] > end_s)
                              | (annotations[right_col] < start_s))].copy()
        
        # Rescale axes to 0.0-1.0 based on location inside chunk
        chunk_annotations.loc[:,[left_col,right_col]] = \
            ((chunk_annotations[[left_col,right_col]] - start_s)
             / (end_s - start_s))
        chunk_annotations.loc[:,[bot_col,top_col]] = \
            (1.0 - ((librosa.hz_to_mel(chunk_annotations[[bot_col,top_col]])
            - freq_axis_low) / (freq_axis_high - freq_axis_low)))
        chunk_annotations = \
            chunk_annotations.loc[chunk_annotations[class_col].isin(classes)]
        trimmed_annots = chunk_annotations.copy()
        trimmed_annots[left_col] = \
            trimmed_annots[left_col].clip(lower=0, upper=1.0)
        trimmed_annots[right_col] = \
            trimmed_annots[right_col].clip(lower=0, upper=1.0)
        trimmed_annots[bot_col] = \
            trimmed_annots[bot_col].clip(lower=0, upper=1.0)
        trimmed_annots[top_col] = \
            trimmed_annots[top_col].clip(lower=0, upper=1.0)
        overlaps = []
        for i in trimmed_annots.index:
            intersection = trimmed_annots.loc[i]
            original = chunk_annotations.loc[i]
            original_area = get_area(original)
            overlaps.append(
                (get_area(intersection)*spec_height*spec_width) / original_area
            )
        chunk_annotations = \
            trimmed_annots.loc[np.array(overlaps) > min_box_percent]
        
        if verbose:
            print(
                "Found {} annotations in chunk".format(len(chunk_annotations))
            )
        
        # Save Chunk as PNG image (lossless compression)
        im = Image.fromarray(mel_spec[::-1, :])
        im = im.convert("L")
        image_filepath = path.join(output_directory, spec_name)
        im.save(image_filepath)
        
        if verbose:
            print("Saved spectrogram to '{}'".format(spec_name))
        
        example_dict = {
            "filepath": image_filepath,
            "height": spec_height,
            "width": spec_width,
            "xmins": trimmed_annots[left_col].tolist(),
            "xmaxs": trimmed_annots[right_col].tolist(),
            "ymins": trimmed_annots[top_col].tolist(),
            "ymaxs": trimmed_annots[bot_col].tolist(),
            "classes_text": trimmed_annots[class_col].tolist(),
            "classes": trimmed_annots[class_col].map(rev_class_map).tolist()
        }
        return example_dict, next_M_init
    
    
    # Actually iterate through the file and extract chunks
    examples = []
    M_init = None
    for ind, start_i in enumerate(start_vals[:-1]):
        spec_name = "{}-{}.png".format(file_id, ind)
        ex, M_init = extract_chunk(
            start_i,
            start_i+chunk_size,
            spec_name,
            M_init=M_init
        )
        examples.append(ex)
    if not drop_last_chunk:
        spec_name = "{}-{}.png".format(file_id, len(start_vals)-1)
        ex, _ = extract_chunk(
            start_vals[-1],
            len(data),
            spec_name,
            M_init=M_init
        )
        examples.append(ex)
    return examples


def generate_dataset(
        splits,
        output_directory,
        label_map_name,
        metadata_name,
        window_size_sec,
        hop_len_sec,
        n_mels,
        freq_max,
        train_chunk_len_sec,
        eval_chunk_len_sec,
        min_box_percent,
        n_train_shards,
        n_eval_shards,
        allowed_classes,
        verbose=False
    ):
    """
    TODO(jwaschura): write detailed documentation for this function
    """
    output_directory = path.abspath(output_directory)
    if verbose:
        print("Output directory is: '{}'".format(output_directory))

    os.makedirs(output_directory)
    if verbose:
        print("Successfully created output directory.")
    
    # Write metadata file to output_directory
    with open(path.join(output_directory, metadata_name), 'w') as metafile:
        json.dump(
            {
                "WINDOW_SIZE_SEC": window_size_sec,
                "HOP_LEN_SEC": hop_len_sec,
                "N_MELS": n_mels,
                "FREQUENCY_MAX": freq_max,
                "TRAIN_CHUNK_SIZE_SEC": train_chunk_len_sec,
                "EVAL_CHUNK_SIZE_SEC": eval_chunk_len_sec,
                "EVAL_CHUNK_STEP_SEC": eval_chunk_len_sec / 2.0,
                "MIN_BOX_PERCENT": min_box_percent,
                "ALLOWED_CLASSES": allowed_classes
            },
            metafile
        )
    
    train_dataset, eval_dataset = get_splits(splits)
    all_annotations = [a for _,a in train_dataset] + [a for _,a in eval_dataset]

    classes = get_all_classes(all_annotations, verbose=True)
    classes = [c for c in classes if c in allowed_classes]
    create_label_map(classes, os.path.join(output_directory, label_map_name))
    
    # The class and reverse class maps are used to encode classes later.
    class_map = {}
    rev_class_map = {}
    for i in range(len(classes)):
        class_map[i+1] = classes[i]
        rev_class_map[classes[i]] = i+1
    
    # Computing scaling parameters based on TRAIN set only.
    min_bound, max_bound = get_minmax_bounds(
        [p[0] for p in train_dataset],
        window_size_sec,
        hop_len_sec,
        n_mels,
        freq_max,
        train_chunk_len_sec
    )
    if verbose:
        print("Min and Max db: {}, {}".format(min_bound, max_bound))

    splits = [
        (
            "train",
            train_dataset,
            n_train_shards,
            train_chunk_len_sec,
            "dense",
            False
        ),
        (
            "eval",
            eval_dataset,
            n_eval_shards,
            eval_chunk_len_sec,
            "sparse",
            True
        )
    ]

    for (splitname,
         dataset,
         num_shards,
         chunk_size,
         chunk_layout,
         drop_last_chunk) in splits:
        examples = []
        for wav_filename, annot_filename in dataset:
            examples.extend(
                process_file(
                    wav_filename,
                    annot_filename,
                    min_bound,
                    max_bound,
                    chunk_size,
                    window_size_sec,
                    hop_len_sec,
                    n_mels,
                    freq_max,
                    min_box_percent,
                    output_directory,
                    classes,
                    rev_class_map,
                    chunk_layout=chunk_layout,
                    drop_last_chunk=drop_last_chunk,
                    verbose=False
                )
            )

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_filebase = os.path.join(
                output_directory,
                "{}_{}".format(splitname, "dataset.record")
            )
            output_tfrecords = \
                tf_record_creation_util.open_sharded_output_tfrecords(
                    tf_record_close_stack,
                    output_filebase,
                    num_shards
                )
            for index, example in enumerate(examples):
                tf_example = create_tf_example(example)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(
                    tf_example.SerializeToString()
                )

# TODO(jwaschura): confirm new script functions as expected and generate new dataset with up-to-date files.
# TODO(jwaschura): add instructions for installing tensorflow object detection.
# TODO(jwaschura): Write documentation for the methods in this file.
