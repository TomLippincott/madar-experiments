import argparse
import pickle
import gzip
import random
import logging
import subprocess
import unicodedata
import re
import string


# Turn users into "A", retweet indicator into "B", and links into "C", otherwise removing non-Arabic characters
def normalize_doc(chars):
    chars = chars.replace("A", "a").replace("B", "b").replace("C", "c")
    chars = re.sub(r"\@\S+?(:?)(\s|$)", r"A\1\2", chars)
    chars = re.sub(r"RT", "B", chars)
    chars = re.sub(r"http(s?)://\S+", r"C", chars)    
    chars = [c for c in chars if any(["ARABIC" in unicodedata.name(c, ""), 
                                      c.isdigit(), 
                                      c in ["A", "B", "C"],
                                      c in string.punctuation, 
                                      c in string.whitespace])]
    retval = "".join(chars)
    return "D" if len(retval) == 0 else retval


def read_data(fname, normalize=False, add_user=False):
    data = []
    with gzip.open(fname, "rt") as ifd:
        for line in ifd:
            toks = line.strip().split("\t")
            if len(toks) == 3:
                uid, label, text = toks[0:3]
            elif len(toks) == 7:
                _, label, text, uid, cid, tlang, scores = toks
            else:
                raise Exception(line)
            text = text.strip()
            if normalize:
                text = normalize_doc(text.strip())
            if add_user:
                text = "{} {}".format(uid, text)
            data.append((text, label))
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", help="Train file")
    parser.add_argument("-n", "--n", dest="n", type=int, help="Maximum ngram length")
    parser.add_argument("-m", "--model", dest="model", help="Model output file")
    parser.add_argument("-l", "--library", dest="library", default="/expscratch/tlippincott/ngram", help="Compiled ngram library location")
    parser.add_argument("--normalize", dest="normalize", default=False, action="store_true", help="Normalize tweets")
    parser.add_argument("--add_user", dest="add_user", default=False, action="store_true", help="Add user")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train = read_data(args.train, args.normalize, args.add_user)
    train_alphabet = set()
    for seq, _ in train:
        for e in seq:
            train_alphabet.add(e)
    logging.info("%d train, %d unique values", len(train), len(train_alphabet))

    train = "\n".join(["{}\t{}\t{}".format(i, l, s) for i, (s, l) in enumerate(train)])
    pid = subprocess.Popen("stack exec -- ngramClassifier train --n {} --modelFile {}".format(args.n, args.model).split(), cwd=args.library, stdin=subprocess.PIPE)
    pid.communicate(train.encode("utf-8"))
