import argparse
import os.path
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    for fname in args.inputs:
        path, f = os.path.split(fname)
        _, task = os.path.split(path)
        f, _ = os.path.splitext(f)
        f, _ = os.path.splitext(f)
        args = f.split("-")
        model, lr, mom, drop, bs, fc, pat, es, ce, we, cs, ws = args
        with open(fname, "rt") as ifd:
            text = ifd.read()
            m = re.match(r".*Pretrained word embeddings covered (\d+)/(\d+) \((.*?)\) of the vocab.*", text, re.I|re.S)
            if m:
                cov = m.group(3)
            else:
                cov = "n/a"
            for m in re.finditer(r"Dev loss/f1/accuracy = (.*?)/(.*?)/(.*?)(\s|$)", text):
                acc = m.group(2)
                f1 = m.group(3)
            
        print("{}\t{}\t{}\t{}\t{}".format(fc, cs, ws, we, f1))

