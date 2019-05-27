import random
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-r", "--reference", dest="reference", help="Reference file")
    parser.add_argument("-f", "--full", dest="full", help="Splits")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    labels = set()
    guesses = {}
    with open(args.input, "rt") as ifd:
        for line in ifd:
            _, guess, text = line.strip().split("\t")[0:3]
            guesses[text] = guess
            labels.add(guess)
    labels = list(labels)
    users = {}
    with open(args.full, "rt") as ifd:
        _ = ifd.readline()
        for line in ifd:
            toks = line.strip().split("\t")
            user = toks[0]
            text = toks[-1]
            if text == "<UNAVAILABLE>":
                continue
            users[user] = users.get(user, {})
            label = guesses[text]
            users[user][label] = users[user].get(label, 0) + 1

    #print(users.keys())


    order = []
    with open(args.reference, "rt") as ifd:    
        _ = ifd.readline()
        for line in ifd:
            user = line.strip().split("\t")[0]
            if user in users:
                guess = sorted(users[user].items(), key=lambda x : x[1], reverse=True)[0][0]
            else:
                random.shuffle(labels)
                guess = labels[0]
            order.append((user, guess))

    with open(args.output, "wt") as ofd:
        ofd.write("\n".join([l for _, l in order]) + "\n")

