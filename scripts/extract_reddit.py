import argparse
import bz2
import lzma
import zstandard as zstd
import json
import os.path
import gzip
import unicodedata
import sys
import re
import io

def zstdOpen(fname, mode):
    with open(fname, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh, read_across_frames=True)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for line in text_stream:
            yield line


# xz bz2 zst
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[-1]
    reader = bz2.open if ext == ".bz2" else lzma.open if ext == ".xz" else zstdOpen if ext == ".zst" else None
    field = "body" if "comments" in args.input else "selftext"
    with gzip.open(args.output, "wt") as ofd:
        if ext == ".zst":
            for line in reader(args.input, "rt"):
                chars = [unicodedata.name(c, "") for c in re.sub(r"\s", "", json.loads(line).get(field, "NULL"))]
                proportion = len([c for c in chars if c.startswith("ARABIC")]) / (len(chars) + 0.001)
                if proportion > 0.5:
                    ofd.write(line)
        else:
            with reader(args.input, "rt") as ifd:
                for line in ifd:
                    chars = [unicodedata.name(c, "") for c in re.sub(r"\s", "", json.loads(line).get(field, "NULL"))]
                    proportion = len([c for c in chars if c.startswith("ARABIC")]) / (len(chars) + 0.001)
                    if proportion > 0.5:
                        ofd.write(line)
