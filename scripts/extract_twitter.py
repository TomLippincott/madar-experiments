import argparse
import gzip
import json
from glob import glob
import os.path
import unicodedata

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd:
        for fname in glob(os.path.join(args.input, "*gz")):
            try:
                with gzip.open(fname, "rt") as ifd:
                    for line in ifd:
                        if line.count("lang\":\"ar") > 1:
                            ofd.write(line)
            except:
                print("Skipping bad file {}".format(fname))
