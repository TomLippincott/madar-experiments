import argparse
import gzip
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-s", "--scored", dest="scored", help="Input file")
    parser.add_argument("-r", "--raw", dest="raw", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    users = []
    langs = []
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            users.append(line.split("\t")[0])

    with open(args.scored, "rt") as ifd:
        for line in ifd:
            langs.append(line.split("\t")[1])

    lookup = {k : v for k, v in zip(users, langs)}
    
    with gzip.open(args.raw, "rt") as ifd, open(args.output, "wt") as ofd:
        for line in ifd:
            _, gold, text, uid, cid, tlang, scores = line.strip().split("\t")
            ofd.write("{}\t{}\t{}\t{}\n".format(gold, lookup[uid], "NA", "NA"))
