from bioacoustic_detection.utils.annot import _format
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import OrderedDict
import math
import numpy as np
import librosa
import librosa.display


def visualize_box_shapes(
        annots,
        bins=100,
        title="Distribution of Box Shapes",
        show=False):
    """
    IN PROGRESS
    """
    
    annot_heights = annots[_format.TOP_COL] - annots[_format.BOT_COL]
    annot_widths = annots[_format.RIGHT_COL] - annots[_format.LEFT_COL]
    plt.hist2d(annot_widths, annot_heights, bins=bins)
    plt.title(title)
    plt.xlabel("Box Width (s)")
    plt.ylabel("Box Height (Hz)")
    if show:
        plt.show()


def plot_annotated_mel_spec(data, samplerate, annotations, cls_col=None, bounds=None, n_fft=1200, hop_length=20,
                            n_mels=400, fmax=1600, adjust_fmax=True, figsize=(15, 5), dpi=300, buffer_s=0.125,
                            title=None, no_axes=False, color_override=None):
    """
    IN PROGRESS
    """
    
    # Extract annotation bounds
    if bounds is None:
        start_s = annotations[_format.LEFT_COL].min() - buffer_s
        end_s = annotations[_format.RIGHT_COL].max() + buffer_s
    else:
        start_s, end_s = bounds
    start_s, end_s = max(start_s, 0.0), min(end_s, len(data)/samplerate)
    observed_max = annotations[_format.TOP_COL].max()
    if adjust_fmax and observed_max > fmax:
        new_fmax = observed_max*1.1
        print("Annotations extend above frequency max of {} Hz, increasing to {:g} Hz.".format(fmax, new_fmax))
        fmax = new_fmax
    shift = math.ceil(n_fft/2)
    start_i, end_i = int(math.floor(start_s*samplerate) - shift), int(math.ceil(end_s*samplerate) + shift)
    if start_i < 0:
        print("Start Index < 0! Setting to 0 instead.")
        start_i = 0
        start_s = (start_i + shift) / samplerate
    if end_i >= len(data):
        print("End Index > length of sequence. Setting to end of sequence instead.")
        end_i = len(data)-1
        end_s = (end_i - shift) / samplerate
    
    # Compute & Draw Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=data[start_i:end_i],
                                              sr=samplerate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=n_mels,
                                              fmax=fmax,
                                              center=False)
    S_dB = librosa.power_to_db(mel_spec, ref=np.max)
    #S_dB,_ = PCEN(mel_spec, mel_spec.shape[1]-1)
    plt.figure(figsize=figsize, dpi=dpi)
    librosa.display.specshow(S_dB,
                             x_axis='time',
                             y_axis='mel',
                             sr=samplerate,
                             hop_length=hop_length,
                             fmax=fmax)
    
    # Draw Annotations
    ax = plt.gca()
    if cls_col is not None:
        classes = annotations[cls_col].unique()
    else:
        classes = ["NA"]
    colors = plt.cm.get_cmap("hsv")
    if color_override is not None:
        class_colors = {classes[c]: color_override for c in range(len(classes))}
    else:
        class_colors = {classes[c]: colors((len(classes)-c) / len(classes)) for c in range(len(classes))}
    print(class_colors)
    for b_i in annotations.index:
        box = annotations.loc[b_i]
        left, right, top, bot = box[_format.LEFT_COL], box[_format.RIGHT_COL], \
                                box[_format.TOP_COL], max(box[_format.BOT_COL], 5)
        if cls_col is not None:
            cls = box[cls_col]
        else:
            cls = "NA"
        
        rect = Rectangle((left - start_s, bot), # X,Y of bottom left
                         right-left, # Width
                         top-bot, # Height
                         linewidth=2,
                         edgecolor=class_colors[cls],
                         facecolor='none',
                         label=cls,
                         alpha=0.7)
        ax.add_patch(rect)
    
    # Decorate Plot
    if no_axes:
        plt.axis('off')
    else:
        y_ticks = [64]
        while y_ticks[-1] * 2 < fmax:
            y_ticks.append(y_ticks[-1] * 2)
        x_ticks = np.linspace(0.0, end_s - start_s, num=5)
        x_tick_labels = ["{:.3f}".format(t) for t in (x_ticks+start_s)]
        plt.xticks(x_ticks, x_tick_labels)
        plt.yticks(y_ticks)
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Frequency (Hz)")
    plt.xlim([0.0, end_s - start_s])
    if title is None:
        plt.title("Mel Spectrogram")
    else:
        plt.title(title)
    if cls_col is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.show()
    plt.close()


# Scans through all annotations and visualizes in 30-second blocks
def visualize_all_annotations(annotations, data, samplerate,
                              block_size=30, step_size=20, n_fft=4096, hop_length=64,
                              n_mels=512, fmax=1600, adjust_fmax=True, figsize=(15, 6)):
    """
    IN PROGRESS
    """
    file_length = len(data) / samplerate
    start = 0
    while start < file_length:
        mask = ~((annotations[_format.LEFT_COL] > start+block_size) | (annotations[_format.RIGHT_COL] < start))
        plot_annotated_mel_spec(data, samplerate,
                                annotations.loc[mask],
                                bounds=[start, start+block_size],
                                buffer_s=0.0,
                                cls_col="Source",
                                figsize=figsize,
                                title="")
        start += step_size