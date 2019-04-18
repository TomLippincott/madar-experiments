import argparse
import re
import torch
import unicodedata
import string
from gensim.models.fasttext import FastText
from torch.utils.data import DataLoader, Dataset
import functools
import numpy
import random
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import logging


class CNN(torch.nn.Module):
    def __init__(self, 
                 filter_count, 
                 char_kernel_widths, 
                 word_kernel_widths, 
                 max_char_length, 
                 max_word_length,
                 char_to_id,
                 word_to_id,
                 label_to_id, 
                 dropout_prob, 
                 character_embeddings,
                 word_embeddings,
                 freeze_character_embeddings,
                 freeze_word_embeddings,
             ):
        super(CNN, self).__init__()

        # set a bunch of dimensions and such
        self._filter_count = filter_count
        self._char_kernel_widths = [c for c in char_kernel_widths if c > 0]
        self._word_kernel_widths = [w for w in word_kernel_widths if w > 0]
        self._max_char_length = max_char_length
        self._max_word_length = max_word_length
        self._nchars = len(char_to_id)
        self._nwords = len(word_to_id)
        self._nlabels = len(label_to_id)
        self._dropout_prob = dropout_prob

        # either initialize uninformed word embeddings of the given size, or read them in
        # from a pretrained embedding model
        try:
            self._word_embedding_size = int(word_embeddings)
            self._word_embeddings = torch.nn.Embedding(num_embeddings=self._nwords,
                                                       embedding_dim=self._word_embedding_size,
                                                       padding_idx=0)
        except:
            embs = FastText.load_fasttext_format(word_embeddings)
            self._word_embedding_size = embs.vector_size
            emb_matrix = numpy.zeros(shape=(self._nwords, self._word_embedding_size))
            found = 0
            for w, i in word_to_id.items():
                if w in embs.wv:
                    found += 1
                emb_matrix[i, :] = embs.wv[w]                    
            self._word_embeddings = torch.nn.Embedding.from_pretrained(torch.Tensor(emb_matrix), freeze=freeze_word_embeddings)
            logging.info("Pretrained word embeddings covered %d/%d (%f) of the vocab", found, self._nwords, found / self._nwords)

        # either initialize uninformed character embeddings of the given size, or read them in
        # from a pretrained embedding model
        try:
            self._char_embedding_size = int(character_embeddings)
            self._char_embeddings = torch.nn.Embedding(num_embeddings=self._nchars,
                                                       embedding_dim=self._char_embedding_size,
                                                       padding_idx=0)
        except:
            embs = FastText.load_fasttext_format(char_embeddings)
            self._char_embedding_size = embs.vector_size
            emb_matrix = numpy.zeros(shape=(self._nchars, self._char_embedding_size))
            found = 0
            for w, i in char_to_id.items():
                if w in embs.wv:
                    found += 1
                emb_matrix[i, :] = embs.wv[w]
            self._char_embeddings = torch.nn.Embedding.from_pretrained(torch.Tensor(emb_matrix), freeze=freeze_character_embeddings)
            logging.info("Pretrained character embeddings covered %d/%d (%f) of the vocab", found, self._nchars, found / self._nchars)

        # character convolutions of each specified size and number of filters
        self.char_convs = torch.nn.ModuleList([torch.nn.Conv1d(1, self._filter_count, (k, self._char_embedding_size)) for k in self._char_kernel_widths])

        # word convolutions of each specified size and number of filters
        self.word_convs = torch.nn.ModuleList([torch.nn.Conv1d(1, self._filter_count, (k, self._word_embedding_size)) for k in self._word_kernel_widths])

        # dropout layer of specified probability
        self.dropout = torch.nn.Dropout(self._dropout_prob)

        # final fully-connected linear layer over the outputs from all the convolutions
        self.output = torch.nn.Linear(len(self._char_kernel_widths + self._word_kernel_widths) * filter_count, 
                                      self._nlabels)

    def forward(self, cx, wx):
        # go from sequences of characters/words to corresponding sequences of embeddings
        cx = self._char_embeddings(cx)
        wx = self._word_embeddings(wx)

        # add a unary dimension, corresponding to the convolutions having a single
        # "input channel" (the embeddings)
        cx = cx.unsqueeze(1)  # (N, Ci, W, D)
        wx = wx.unsqueeze(1)

        # apply the convolutions and max pooling
        cx = [F.relu(conv(cx)).squeeze(3) for conv in self.char_convs]
        cx = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cx]
        wx = [F.relu(conv(wx)).squeeze(3) for conv in self.word_convs]
        wx = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in wx]
        cx = [torch.cat(cx, 1)] if len(cx) > 0 else []
        wx = [torch.cat(wx, 1)] if len(wx) > 0 else []
        x = torch.cat(cx + wx, 1)
        x = self.dropout(x)
        logit = self.output(x)
        return torch.nn.functional.log_softmax(logit, dim=1)
        

# simple dataset class, nothing really DID-specific...
class DidData(Dataset):
    def __init__(self, items):
        super(Dataset, self).__init__()
        self._items = items
    def __len__(self):
        return len(self._items)
    def __getitem__(self, i):
        return self._items[i]


# callback for turning a list of (chars, words, label) triplets into
# minibatch tensors, according to char, word, and label lookups
def collate(clu, wlu, llu, max_char_length, max_word_length, gpu, triplets):
    y = numpy.zeros(shape=(len(triplets), len(llu)))
    for i, pair in enumerate(triplets):
        y[i, llu[pair[2]]] = 1.0
    wx = numpy.zeros(shape=(len(triplets), max_word_length))
    cx = numpy.zeros(shape=(len(triplets), max_char_length))
    for i, pair in enumerate(triplets):
        chars = [clu.get(c, 1) for c in pair[0]][0:max_char_length]
        cx[i, 0:len(chars)] = chars
        words = [wlu.get(c, 1) for c in pair[1]][0:max_word_length]
        wx[i, 0:len(words)] = words
    return [(x.cuda() if gpu else x) for x in [torch.LongTensor(cx), torch.LongTensor(wx), torch.Tensor(y)]]
    
        
# Turn users into "A", retweet indicator into "B", and links into "C", otherwise removing non-Arabic/punc/ws
def normalize_doc(chars):
    if "<UNAVAILABLE>" in chars:
        return None
    chars = chars.replace("A", "a").replace("B", "b").replace("C", "c")
    chars = re.sub(r"\@\S+?(:?)(\s|$)", r"A\1\2", chars)
    chars = re.sub(r"RT", "B", chars)
    chars = re.sub(r"http(s?)://\S+", r"C", chars)    
    chars = [c for c in chars if any(["ARABIC" in unicodedata.name(c, ""), 
                                      c.isdigit(), 
                                      c in ["A", "B", "C"],
                                      c in string.punctuation, 
                                      c in string.whitespace])]
    return "".join(chars)


# Run model on given data loader, return (loss, macro F1, accuracy)
def evaluate(loader, model, loss_metric):
    model.eval()
    total_loss = 0.0
    guesses = []
    gold = []
    for i, (cx, wx, y) in enumerate(loader, 1):
        out = model(cx, wx)
        loss = torch.mean(loss_metric(out, y))
        guesses.append([x.item() for x in out.argmax(1)])
        gold.append([x.item() for x in y.argmax(1)])
        total_loss += float(loss)
    gold = sum(gold, [])
    guesses = sum(guesses, [])
    items = [loader.dataset[i][0] for i in range(len(loader.dataset))]
    macro_f1 = f1_score(gold, guesses, average="macro")
    acc = accuracy_score(gold, guesses)
    model.train()
    return (total_loss, macro_f1, acc, zip(items, gold, guesses))


# Recursively initialize model weights
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", 
                        dest="train", 
                        help="Train file")
    parser.add_argument("--train_count", 
                        dest="train_count", 
                        type=int, 
                        help="Number of train instances (default=all)")
    parser.add_argument("--dev", 
                        dest="dev", 
                        help="Dev file")
    parser.add_argument("--dev_count", 
                        dest="dev_count", 
                        type=int, 
                        help="Number of dev instances (default=all)")
    parser.add_argument("--character_embeddings", 
                        dest="character_embeddings", 
                        help="Integer or pretrained model")
    parser.add_argument("--word_embeddings", 
                        dest="word_embeddings", 
                        help="Integer or pretrained model")
    parser.add_argument("--freeze_character_embeddings", 
                        dest="freeze_character_embeddings", 
                        action="store_true", 
                        default=False, 
                        help="Don't adapt pretrained character embeddings during training")
    parser.add_argument("--freeze_word_embeddings", 
                        dest="freeze_word_embeddings", 
                        action="store_true", 
                        default=False, 
                        help="Don't adapt pretrained word embeddings during training")
    parser.add_argument("--dropout", 
                        dest="dropout", 
                        default=0.5, 
                        type=float, 
                        help="Dropout probability on final layer")
    parser.add_argument("--batch_size", 
                        dest="batch_size", 
                        type=int, 
                        default=32, 
                        help="Batch size")
    parser.add_argument("--max_char_length", 
                        dest="max_char_length", 
                        type=int, 
                        help="Maximum sequence length for padding/truncation")
    parser.add_argument("--max_word_length", 
                        dest="max_word_length", 
                        type=int, 
                        help="Maximum sequence length for padding/truncation")
    parser.add_argument("--char_kernel_sizes", 
                        dest="char_kernel_sizes", 
                        default="1,2,3,4,5", 
                        help="Sizes of convolutional kernels (1,2,3,4,5)")
    parser.add_argument("--word_kernel_sizes", 
                        dest="word_kernel_sizes", 
                        default="1,2,3,4,5", 
                        help="Sizes of convolutional kernels (1,2,3,4,5)")
    parser.add_argument("--filter_count", 
                        dest="filter_count", 
                        type=int, 
                        default=100, 
                        help="Number of filters per convolution")
    parser.add_argument("--gpu", 
                        dest="gpu", 
                        default=False, 
                        action="store_true", 
                        help="Run on GPU")
    parser.add_argument("--normalize", 
                        dest="normalize", 
                        default=False, 
                        action="store_true", 
                        help="Normalize to remove non-Arabic characters etc")
    parser.add_argument("--learning_rate", 
                        dest="learning_rate", 
                        default=0.1, 
                        type=float, 
                        help="Initial SGD learning rate")
    parser.add_argument("--momentum", 
                        dest="momentum", 
                        default=0.9, 
                        type=float, 
                        help="SGD momentum")
    parser.add_argument("--patience", 
                        dest="patience", 
                        type=int, 
                        default=10, 
                        help="Number of epochs for patience in learning rate reduction")
    parser.add_argument("--early_stop", 
                        dest="early_stop", 
                        type=int, 
                        default=20, 
                        help="Number of epochs for early stop")
    parser.add_argument("--epochs", 
                        dest="epochs", 
                        type=int, 
                        default=1000, 
                        help="Maximum number of epochs")
    parser.add_argument("--seed", 
                        dest="seed", 
                        type=int, 
                        help="Random seed (use system state if unspecified)")
    parser.add_argument("--output", 
                        dest="output", 
                        help="Output file")
    parser.add_argument("--log_level", 
                        dest="log_level", 
                        default="DEBUG",
                        choices=["DEBUG, ""INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    train, dev = [], []

    normalize = False
    max_char_length, max_word_length = 0, 0
    total = 0
    with open(args.train, "rt") as ifd:
        line = ifd.readline()
        try:
            _, _, _, _, _, _ = line.strip().split("\t")
        except:
            char_seq, label = line.strip().split("\t")
            word_seq = char_seq.split()
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            train.append((char_seq, word_seq, label))
        for line in ifd:
            try:
                _, _, _, _, label, char_seq = line.strip().split("\t")
            except:
                char_seq, label = line.strip().split("\t")
            if args.normalize:
                char_seq = normalize_doc(char_seq)
            if char_seq == None:
                continue
            word_seq = char_seq.split()
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            train.append((char_seq, word_seq, label))

    random.shuffle(train)
    train = train[0:len(train) if args.train_count == None else args.train_count]


    label_to_id = {} #"<UNK>" : 0}
    char_to_id = {"<PAD>" : 0, "<UNK>" : 1}
    word_to_id = {"<PAD>" : 0, "<UNK>" : 1}
    for (cs, ws, l) in train:
        for c in cs:
            char_to_id[c] = char_to_id.get(c, len(char_to_id))
        for w in ws:
            word_to_id[w] = word_to_id.get(w, len(word_to_id))
        label_to_id[l] = label_to_id.get(l, len(label_to_id))

    
    with open(args.dev, "rt") as ifd:
        line = ifd.readline()
        try:
            _, _, _, _, _, _ = line.strip().split("\t")
        except:
            char_seq, label = line.strip().split("\t")
            word_seq = char_seq.split()
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            dev.append((char_seq, word_seq, label))
        for line in ifd:
            try:
                _, _, _, _, label, char_seq = line.strip().split("\t")
            except:
                char_seq, label = line.strip().split("\t")
            if args.normalize:
                char_seq = normalize_doc(char_seq)
            if char_seq == None:
                continue
            word_seq = char_seq.split()
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            dev.append((char_seq, word_seq, label))

    dev = dev[0:len(dev) if args.dev_count == None else args.dev_count]

    for (cs, ws, l) in dev:
        for c in cs:
            char_to_id[c] = char_to_id.get(c, len(char_to_id))
        for w in ws:
            word_to_id[w] = word_to_id.get(w, len(word_to_id))
        label_to_id[l] = label_to_id.get(l, len(label_to_id))
    id_to_label = {v : k for k, v in label_to_id.items()}


    logging.info("Loaded %d train and %d dev instances, with %d features and %d labels, maximum character length %d, max word length %d", len(train), len(dev), len(char_to_id), len(label_to_id), max_char_length, max_word_length)
    
    train_loader = DataLoader(DidData(train), 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=functools.partial(collate, 
                                                           char_to_id, 
                                                           word_to_id,
                                                           label_to_id, 
                                                           max_char_length, 
                                                           max_word_length,
                                                           args.gpu))
    dev_loader = DataLoader(DidData(dev), 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            collate_fn=functools.partial(collate, 
                                                           char_to_id, 
                                                           word_to_id,
                                                           label_to_id, 
                                                           max_char_length, 
                                                           max_word_length,
                                                           args.gpu))

    model = CNN(args.filter_count, 
                list(map(int, args.char_kernel_sizes.split(","))),
                list(map(int, args.word_kernel_sizes.split(","))),
                max_char_length, 
                max_word_length, 
                char_to_id, 
                word_to_id,
                label_to_id,
                args.dropout,
                args.character_embeddings,
                args.word_embeddings,
                args.freeze_character_embeddings,
                args.freeze_word_embeddings,
            )

    # if GPU is to be used
    if args.gpu:
        model.cuda()

    # recursively initialize model weights
    model.apply(init_weights)
    metric = torch.nn.KLDivLoss(reduction="batchmean")
    optim = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    sched = ReduceLROnPlateau(optim, patience=args.patience, verbose=True)
    
    best_dev_loss = None
    best_dev_out = None
    since_improvement = 0
    for epoch in range(1, args.epochs):
        loss_total = 0.0
        for i, batch in enumerate(train_loader):
            cx, wx, y = batch
            model.zero_grad()            
            out = model(cx, wx)
            loss = metric(out, y)
            loss = torch.mean(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_total += loss

        dev_loss, dev_f1, dev_acc, dev_out = evaluate(dev_loader, model, metric)
        sched.step(dev_loss)
        logging.info("Epoch %.4d: Train loss = %.3f\tDev loss/f1/accuracy = %.3f/%.3f/%.3f", epoch, loss_total, dev_loss, dev_f1, dev_acc)
        if best_dev_loss == None or dev_loss < best_dev_loss:
            since_improvement = 0
            best_dev_loss = dev_loss
            best_dev_out = dev_out
        else:
            since_improvement += 1
            if since_improvement > args.early_stop:
                logging.info("Stopping early after %d epochs with no improvement", args.early_stop)
                break
                
    with open(args.output, "wt") as ofd:
        for item, gold, guess in best_dev_out:
            ofd.write("{}\t{}\t{}\n".format(id_to_label[guess], id_to_label[gold], item))
