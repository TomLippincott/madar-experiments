import argparse
import gzip
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    users = {}
    langs = {}
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            _, label, text, uid, cid, tlang, scores = line.strip().split("\t")
            if scores == "<NIL>":
                scores = None
            else:
                scores = [float(x) for x in scores.split(",")]
            users[uid] = users.get(uid, []) + [text]
            langs[uid] = label
    with gzip.open(args.output, "wt") as ofd:
        for i, (uid, texts) in enumerate(users.items()):
            ofd.write("{}\t{}\t{}\n".format(uid, langs[uid], re.sub("\s+", " ", "  ".join(texts))))
