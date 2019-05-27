import argparse
import os.path
import re
import gzip
import pickle
import numpy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    items = []
    fields = set()
    for fname in args.inputs:
        with gzip.open(fname, "rb") as ifd:
            labels, score, conf = pickle.load(ifd)
        config = {k : v for k, v in [x.split("=") for x in re.match(r"^(.*).pkl.gz$", os.path.basename(fname)).group(1).split("-")]}
        items.append((config, score))
        for k in config.keys():
            fields.add(k)
    fields = list(sorted(fields))
    with gzip.open(args.output, "wt") as ofd:
        ofd.write("\t".join(fields + ["F1"]) + "\n")
        for item, score in items:
            ofd.write("\t".join([item.get(f, "NA") for f in fields] + [str(score)]) + "\n")
