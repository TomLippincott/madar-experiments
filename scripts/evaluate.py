import argparse
import pickle
from sklearn.metrics import confusion_matrix, f1_score
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    golds = []
    guesses = []
    with open(args.input, "rt") as ifd:
        for line in ifd:
            gold, guess, _, _ = line.split("\t")
            golds.append(gold)
            guesses.append(guess)
    labels = list(sorted(list(set(golds))))
    score = f1_score(y_true=golds, y_pred=guesses, average="macro")
    conf = confusion_matrix(y_true=golds, y_pred=guesses, labels=labels)
    with gzip.open(args.output, "wb") as ofd:
        pickle.dump((labels, score, conf), ofd)
