import argparse
import os.path
import re
import gzip
import pickle
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.input, "rb") as ifd:
        classes, score, conf = pickle.load(ifd)
    acc = conf.diagonal().sum() / conf.sum()
    title = "F-score: %.3f, Acc: %.3f" % (score, acc)
    cm = conf.astype('float') / conf.sum(axis=1)[:, numpy.newaxis]
    cmap=plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=8)

    plt.setp(ax.get_yticklabels(), fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output)
