import gzip
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
import sys
import valid

# PREDICT AT USER LEVEL

class CNN(torch.nn.Module):
    def __init__(self,
                 feat_to_id,
                 max_length,
                 embeddings,
                 freeze_embeddings,
                 filter_count, 
                 kernel_widths,
                 output_size,
             ):
        super(CNN, self).__init__()

        # set a bunch of dimensions and such
        self._standalone = True
        self._output_size = output_size
        self._filter_count = filter_count
        self._kernel_widths = [c for c in kernel_widths]
        self._max_length = max_char_length
        self._nfeats = len(feat_to_id)
        #self._dropout_prob = dropout_prob
        #self._output_size = output_size
        
        # either initialize uninformed embeddings of the given size, or read them in
        # from a pretrained embedding model
        try:
            self._embedding_size = int(embeddings)
            self._embeddings = torch.nn.Embedding(num_embeddings=self._nfeats,
                                                  embedding_dim=self._embedding_size,
                                                  padding_idx=0)
        except:
            embs = FastText.load_fasttext_format(embeddings)
            self._embedding_size = embs.vector_size
            emb_matrix = numpy.zeros(shape=(self._nfeats, self._embedding_size))
            found = 0
            for f, i in feat_to_id.items():
                if f in embs.wv:
                    found += 1
                emb_matrix[i, :] = embs.wv[w]                    
            self._embeddings = torch.nn.Embedding.from_pretrained(torch.Tensor(emb_matrix), freeze=freeze_embeddings)
            logging.info("Pretrained embeddings covered %d/%d (%f) of the vocab", found, self._nfeats, found / self._nfeats)

        # convolutions of each specified size and number of filters
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(1, self._filter_count, (k, self._embedding_size)) for k in self._kernel_widths])
        self.output = torch.nn.Linear(self.penultimate_size, self.output_size)
        
    @property
    def penultimate_size(self):
        return len(self._kernel_widths) * self._filter_count

    @property
    def output_size(self):
        return self._output_size
        
    def forward(self, x):
        # go from sequences of characters/words to corresponding sequences of embeddings
        x = self._embeddings(x)

        # add a unary dimension, corresponding to the convolutions having a single
        # "input channel" (the embeddings)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        # apply the convolutions and max pooling
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = [torch.cat(x, 1)] if len(x) > 0 else []
        x = torch.cat(x, 1)
        return (F.log_softmax(self.output(x), dim=1) if self._standalone else x)


class RNN(torch.nn.Module):
    def __init__(self, 
                 feat_to_id,
                 max_length, 
                 embeddings,
                 freeze_embeddings,
                 hidden_size,
                 bidirectional,
                 num_layers,
                 dropout,
                 output_size,
             ):
        super(RNN, self).__init__()
        self._standalone = True
        self._max_length = max_char_length
        self._nfeats = len(feat_to_id)
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers
        self._output_size = output_size
        self._dropout = dropout
        try:
            self._embedding_size = int(embeddings)
            self._embeddings = torch.nn.Embedding(num_embeddings=self._nfeats,
                                                  embedding_dim=self._embedding_size,
                                                  padding_idx=0)
        except:
            embs = FastText.load_fasttext_format(embeddings)
            self._embedding_size = embs.vector_size
            emb_matrix = numpy.zeros(shape=(self._nfeats, self._embedding_size))
            found = 0
            for f, i in feat_to_id.items():
                if f in embs.wv:
                    found += 1
                emb_matrix[i, :] = embs.wv[w]                    
            self._embeddings = torch.nn.Embedding.from_pretrained(torch.Tensor(emb_matrix), freeze=freeze_embeddings)
            logging.info("Pretrained embeddings covered %d/%d (%f) of the vocab", found, self._nfeats, found / self._nfeats)
        self.rnn = torch.nn.LSTM(input_size=self._embedding_size,
                                 hidden_size=self._hidden_size,
                                 num_layers=self._num_layers,
                                 batch_first=True,
                                 bidirectional=self._bidirectional,
                                 dropout=self._dropout,
        )
        self.output = torch.nn.Linear(self.penultimate_size, self.output_size)

    @property
    def penultimate_size(self):
        return (2 if self._bidirectional else 1) * self._hidden_size

    @property
    def output_size(self):
        return self._output_size
        
    def forward(self, x):
        x = self._embeddings(x)
        out, (h, _) = self.rnn(x)
        out, _ = torch.max(out, 1)
        return (F.log_softmax(self.output(out), dim=1) if self._standalone else out)


class MLP(torch.nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size_list,
                 output_size,
             ):
        super(MLP, self).__init__()
        self._standalone = True
        self._output_size = output_size
        self._size_list = [input_size] + hidden_size_list
        self.layers = torch.nn.ModuleList()
        for i in range(len(self._size_list) - 1):
            self.layers.append(torch.nn.Linear(self._size_list[i], self._size_list[i + 1])) 
        self.output = torch.nn.Linear(self.penultimate_size, self.output_size)
            
    def forward(self, x):
        for i in range(len(self._size_list) - 1):
            x = F.relu(self.layers[i](x))
        return F.log_softmax(self.output(x), dim=1) if self._standalone else x

    @property
    def output_size(self):
        return self._output_size

    @property
    def penultimate_size(self):
        return self._size_list[-1]

    
class Ensemble(torch.nn.Module):
    def __init__(self,
                 gold_to_id,
                 label_to_id,
                 dist_sizes,
                 feats_to_ids,
                 max_lengths,
                 embeddings,
                 freeze,
                 dropout,
                 
                 filter_counts, 
                 kernel_widths, 

                 rnn_hidden_sizes,
                 rnn_dropout,
                 standalone=False,
                 #mlp_hidden_size_lists,
             ):
        super(Ensemble, self).__init__()
        self._standalone = standalone
        self._model = 0
        self._output_size = len(gold_to_id)
        self._dropout = dropout
        self._rnn_dropout = rnn_dropout
        self.linears = torch.nn.ModuleList()
        self.cnn_submodels = torch.nn.ModuleList()
        for f2i, ml, emb, fr, fc, kw in zip(feats_to_ids, max_lengths, embeddings, freeze, filter_counts, kernel_widths):
            if len(kw) > 0:
                self.cnn_submodels.append(CNN(f2i, ml, emb, fr, fc, kw, self._output_size))
                #self.linears.append(torch.nn.Linear(self.cnn_submodels[-1].output_size, self._output_size))
        self.rnn_submodels = torch.nn.ModuleList()
        for f2i, ml, emb, fr, hidden, do in zip(feats_to_ids, max_lengths, embeddings, freeze, rnn_hidden_sizes, rnn_dropout):
            if hidden > 0:
                self.rnn_submodels.append(RNN(f2i, ml, emb, fr, hidden, True, 2, do, self._output_size))
                #self.linears.append(torch.nn.Linear(self.rnn_submodels[-1].out_size, self._output_size))
        self.dist_submodels = torch.nn.ModuleList()
        for i in range(len(dist_sizes)):
            self.dist_submodels.append(MLP(dist_sizes[i], [32], self._output_size))
            #self.linears.append(torch.nn.Linear(self.dist_submodels[-1].output_size, self._output_size))
        self.label_submodels = torch.nn.ModuleList()
        for i in range(len(label_to_id)):
            self.label_submodels.append(MLP(len(label_to_id[i]), [32], self._output_size))

        self._combined_size = 0
        for xs in [self.cnn_submodels, self.dist_submodels, self.label_submodels, self.rnn_submodels]:
            for sm in xs:
                self._combined_size += sm.output_size if standalone else sm.penultimate_size
                #if standalone == True:
                #    self._combined_size += sm.output_size #self._output_size
                #else:
                #    pass
        self.fcA = torch.nn.Linear(self._combined_size, 256)
        self.fcB = torch.nn.Linear(256, 128)
        self.linear = torch.nn.Linear(128, self._output_size)

            #self.linears.append(torch.nn.Linear(self.label_submodels[-1].out_size, self._output_size))
        #self.fc = torch.nn.Linear(len(self.linears) * self._output_size, len(self.linears) * self._output_size)
        #self.fc = torch.nn.Linear(len(self.linears) * self._output_size, self._output_size)
        #self._combined_size = sum([x.out_size for x in self.cnn_submodels] + [x.out_size for x in self.rnn_submodels] + [x.out_size for x in self.dist_submodels] + [x.out_size for x in self.label_submodels])
        self.dropout = torch.nn.Dropout(self._dropout)

        #self.first = torch.nn.Linear(self._combined_size, 512)
        #self.second = torch.nn.Linear(512, 256)
        #self.linear = torch.nn.Linear(self._output_size, self._output_size)


    def forward(self, sequence_data, dists, labelings):
        outputs = [] #self.dist_submodels[2](dists[2])]
        for submodel, data in zip(self.cnn_submodels, sequence_data):
            submodel._standalone = self._standalone
            outputs.append(submodel(data))
        for submodel, data in zip(self.rnn_submodels, sequence_data):
            submodel._standalone = self._standalone
            outputs.append(submodel(data))
        for submodel, data in zip(self.dist_submodels, dists):
            submodel._standalone = self._standalone
            outputs.append(submodel(data))
        for submodel, data in zip(self.label_submodels, labelings):
            submodel._standalone = self._standalone
            outputs.append(submodel(data))
        #print([x.shape for x in outputs])
        #sys.exit()
        #transforms = []
        #for output, linear in zip(outputs, self.linears):
        #    transforms.append(F.softmax(linear(output), dim=1))
        #print(outputs[0].shape)
        one = torch.cat(outputs, 1) # batch * submodel * labels
        #print(one.shape)
        #sys.exit()
        one = self.dropout(one)

        #sh = transforms.shape
        one = F.relu(self.fcA(one))
        one = F.relu(self.fcB(one))
        #one = self.fc(one)
        
        return torch.nn.functional.log_softmax(self.linear(one), dim=1)

    
        
# simple dataset class, nothing really DID-specific...
class DidData(Dataset):
    def __init__(self, items):
        super(Dataset, self).__init__()
        self._items = items
    def __len__(self):
        return len(self._items)
    def __getitem__(self, i):
        return self._items[i]


# callback for turning a list of (chars, words, labelings, distributions, gold) 
# tuples into minibatch tensors, according to char, word, and label lookups
def collate(clu, wlu, llu, ds, glu, max_char_length, max_word_length, gpu, tuples):
    y = numpy.zeros(shape=(len(tuples), len(glu)))
    #tl = numpy.zeros(shape=(len(tuples), len(tlu)))
    #sc = numpy.zeros(shape=(len(tuples), 26))
    for i, (_, _, _, _, l) in enumerate(tuples):
        if l in glu:
            y[i, glu[l]] = 1.0
        #tl[i, tlu.get(t, 0)] = 1.0
        #sc[i, :] = s
    wx = numpy.zeros(shape=(len(tuples), max_word_length))
    cx = numpy.zeros(shape=(len(tuples), max_char_length))
    for i, (_chars, _words, _, _, _) in enumerate(tuples):
        chars = [clu.get(c, 1) for c in _chars][0:max_char_length]
        cx[i, 0:len(chars)] = chars
        words = [wlu.get(c, 1) for c in _words][0:max_word_length]
        wx[i, 0:len(words)] = words
        
    dists = []
    for i in range(len(tuples[0][3])):
        dists.append(numpy.zeros(shape=(len(tuples), ds[i])))
    for i, (_, _, _, d, _) in enumerate(tuples):
        for j, dist in enumerate(d):
            dists[j][i, :] = dist
    dists = [torch.Tensor(x) for x in dists]
    dists = [(x.cuda() if gpu else x) for x in dists]
        
    labelings = []
    for i in range(len(tuples[0][2])):
        labelings.append(numpy.zeros(shape=(len(tuples), len(llu[i]))))
    for i, (_, _, l, _, _) in enumerate(tuples):
        for j, label in enumerate(l):
            labelings[j][i, llu[j][label]] = 1.0
    labelings = [torch.Tensor(x) for x in labelings]
    labelings = [(x.cuda() if gpu else x) for x in labelings]

    cx, wx, y = [(x.cuda() if gpu else x) for x in [torch.LongTensor(cx), torch.LongTensor(wx), torch.Tensor(y)]]
    return ([cx, wx], dists, labelings, y)
    
        
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
def evaluate_model(loader, model, loss_metric, name=None, a=None, b=None):
    model.eval()
    total_loss = 0.0
    guesses = []
    gold = []
    losses = []
    total_items = 0
    if a == None:
        for i, (seq, dists, labelings, y) in enumerate(loader, 1):
            out = model(seq, dists,labelings)
            loss = torch.mean(loss_metric(out, y))
            guesses.append([j.item() for j in out.argmax(1)])
            gold.append([j.item() for j in y.argmax(1)])
            total_loss += float(loss) * y.shape[0]
            total_items += y.shape[0]
    else:
        sm = getattr(model, name)[b]
        old = sm._standalone
        sm._standalone = True
        for i, tpl in enumerate(loader, 1):
            x = tpl[a][b]
            y = tpl[3]
            out = getattr(model, name)[b](x)
            loss = torch.mean(loss_metric(out, y))
            guesses.append([j.item() for j in out.argmax(1)])
            gold.append([j.item() for j in y.argmax(1)])
            total_loss += float(loss) * y.shape[0]
            total_items += y.shape[0]
        sm._standalone = old
    gold = sum(gold, [])
    guesses = sum(guesses, [])
    items = [loader.dataset[i][0] for i in range(len(loader.dataset))]
    macro_f1 = f1_score(gold, guesses, average="macro")
    acc = accuracy_score(gold, guesses)
    model.train()
    return (total_loss / total_items, macro_f1, acc, zip(items, gold, guesses))


def apply_model(loader, model, loss_metric):
    model.eval()
    total_loss = 0.0
    guesses = []
    gold = []
    # Saudi_Arabia = default
    for i, (seq, dists, labelings, y) in enumerate(loader, 1):
        out = model(seq, dists, labelings)
        #out = model(seq[0], seq[1])
        #loss = torch.mean(loss_metric(out, y))
        guesses.append([x.item() for x in out.argmax(1)])
        #gold.append([x.item() for x in y.argmax(1)])
        #total_loss += float(loss)
    #gold = sum(gold, [])
    guesses = sum(guesses, [])
    items = [loader.dataset[i][0] for i in range(len(loader.dataset))]
    #macro_f1 = f1_score(gold, guesses, average="macro")
    #acc = accuracy_score(gold, guesses)
    model.train()
    return list(zip(items, guesses))


# Recursively initialize model weights
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def read_data(fname, normalize=False, add_user=False):
    data = []
    with gzip.open(fname, "rt") as ifd:
        for line in ifd:
            toks = line.strip().split("\t")
            if len(toks) == 3:
                uid, label, text = toks[0:3]
                scores = False
            elif len(toks) == 7:
                _, label, text, uid, cid, tlang, scores = toks
                if scores == "<NIL>":
                    scores = None
                else:
                    scores = [float(x) for x in scores.split(",")]
            else:
                raise Exception(line)
            text = text.strip()
            if normalize:
                text = normalize_doc(text.strip())
            if add_user:
                text = "{} {}".format(uid, text)
            if scores == False:
                data.append((text, text.split(), label))
            else:
                data.append((text, text.split(), label, uid, tlang, scores))
    logging.info("Read %d lines from %s", len(data), fname)
    return data


def exp_normalize(x):
    xx = numpy.asarray([v[0] for v in x])
    b = xx.max()
    y = numpy.exp(xx - b)
    vv = y / y.sum()
    return vv.tolist()


def read_lm(fname, num):
    data = []
    with open(fname, "rt") as ifd:
        for line in ifd:
            gold, guess, text, scores = line.strip().split("\t")
            scores = exp_normalize([(float(x.split("=")[1]), x.split("=")[0]) for x in sorted(scores.split())])
            data.append(scores)
    logging.info("Read %d lines from %s", len(data), fname)
    assert(len(data) == num)
    return data


def read_instances(fname, lms, args, collapse_users=False):
    insts = read_data(fname, args.normalize, args.add_user)
    if len(insts[0]) == 6:
        users = [i[3] for i in insts]
        twitter_labeling = [[i[4] for i in insts]]
        sh = len(insts[0][5]) #.shape[0]
        madar_scores = [[(numpy.asarray([0.0 for j in range(sh)]) if i[-1] == None else i[-1]) for i in insts]]
    else:
        twitter_labeling = []
        madar_scores = []
    char_seqs = [i[0] for i in insts]
    word_seqs = [i[1] for i in insts]
    golds = [i[2] for i in insts]
    dummy = [() for i in range(len(insts))]
    distributions = list(zip(*([read_lm(x, len(insts)) for x in lms] + madar_scores))) if len(lms) > 0 else dummy
    labelings = list(zip(*twitter_labeling)) if len(twitter_labeling) != 0 else dummy
    return list(zip(*[x for x in [char_seqs, word_seqs, labelings, distributions, golds] if len(x) > 0]))


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
    parser.add_argument("--test", 
                        dest="test", 
                        help="Test file")
    parser.add_argument("--character_embeddings", 
                        dest="character_embeddings",
                        default=100,
                        help="Integer or pretrained model")
    parser.add_argument("--word_embeddings", 
                        dest="word_embeddings",
                        default=100,
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
    parser.add_argument("--rnn_dropout", 
                        dest="rnn_dropout", 
                        default=0.0, 
                        type=float, 
                        help="Dropout probability on RNNs")
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
                        help="Sizes of convolutional kernels, e.g. 1,2,3")
    parser.add_argument("--word_kernel_sizes", 
                        dest="word_kernel_sizes", 
                        help="Sizes of convolutional kernels, e.g. 1,2,3")
    parser.add_argument("--char_rnn_hidden_size", 
                        dest="char_rnn_hidden_size",
                        type=int,
                        default=0,
                        help="Size of character RNN hidden size")
    parser.add_argument("--word_rnn_hidden_size", 
                        dest="word_rnn_hidden_size",
                        type=int,
                        default=0,
                        help="Size of word RNN hidden size")                    
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
                        help="Normalize tweet to remove non-Arabic characters etc")
    parser.add_argument("--add_user", 
                        dest="add_user", 
                        default=False, 
                        action="store_true", 
                        help="Add user name to tweet")
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
    parser.add_argument("--pretrain_epochs", 
                        dest="pretrain_epochs", 
                        type=int, 
                        default=1000, 
                        help="Maximum number of epochs")
    parser.add_argument("--seed", 
                        dest="seed", 
                        type=int, 
                        help="Random seed (use system state if unspecified)")
    parser.add_argument("--dev_output", 
                        dest="dev_output", 
                        help="Dev output file")
    parser.add_argument("--test_output", 
                        dest="test_output", 
                        help="Test output file")
    parser.add_argument("--freeze_submodels",
                        dest="freeze_submodels",
                        default=False,
                        action="store_true")
    parser.add_argument("--log_level", 
                        dest="log_level", 
                        default="DEBUG",
                        choices=["DEBUG, ""INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument(dest="lms", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # instance: chars, words, gold, [labels], [scores]
    train_insts = read_instances(args.train, [x for x in sorted(args.lms) if "train" in x], args)
    dev_insts = read_instances(args.dev, [x for x in sorted(args.lms) if "dev" in x], args)
    test_insts = read_instances(args.test, [x for x in sorted(args.lms) if "test" in x], args)

    #train_insts = [x for x in train_insts if x[4] != "MSA"]
    #dev_insts = [x for x in dev_insts if x[4] != "MSA"]
    #test_insts = [x for x in test_insts if x[4] != "MSA"]

    #print(train_insts[0])
    #print(len(train_insts), len(dev_insts), len(test_insts))
    #sys.exit()

    gold_to_id = {} #"<UNK>" : 0}
    label_to_id = {}
    char_to_id = {"<PAD>" : 0}
    word_to_id = {"<PAD>" : 0}
    dist_sizes = {}
    max_char_length = 0
    max_word_length = 0
    for chars, words, labels, dists, gold in train_insts + dev_insts + test_insts:
        max_char_length = max(max_char_length, len(chars))
        for c in chars:
            char_to_id[c] = char_to_id.get(c, len(char_to_id))
        max_word_length = max(max_word_length, len(words))
        for w in words:
            word_to_id[w] = word_to_id.get(w, len(word_to_id))
        if gold != "UNK":
            gold_to_id[gold] = gold_to_id.get(gold, len(gold_to_id))
        for i, v in enumerate(labels):
            label_to_id[i] = label_to_id.get(i, {})
            label_to_id[i][v] = label_to_id[i].get(v, len(label_to_id[i]))
        for i, v in enumerate(dists):        
            dist_sizes[i] = max(dist_sizes.get(i, 0), len(v))

    label_feats = sum([len(x) for x in label_to_id.values()])
    dist_feats = sum(dist_sizes.values())
    logging.info("Loaded %d train, %d dev, and %d test instances, with %d/%d unique characters/words and %d gold labels, maximum character sequence length %d, maximum word sequence length %d, %d features from labelings, %d features from distributions", len(train_insts), len(dev_insts), len(test_insts), len(char_to_id) - 1, len(word_to_id) - 1, len(gold_to_id), max_char_length, max_word_length, label_feats, dist_feats)

    train_loader = DataLoader(DidData(train_insts), 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=functools.partial(collate, 
                                                           char_to_id, 
                                                           word_to_id,
                                                           label_to_id,
                                                           dist_sizes,
                                                           gold_to_id,
                                                           max_char_length, 
                                                           max_word_length,
                                                           args.gpu))
    dev_loader = DataLoader(DidData(dev_insts), 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            collate_fn=functools.partial(collate, 
                                                         char_to_id, 
                                                         word_to_id,
                                                         label_to_id,
                                                         dist_sizes,
                                                         gold_to_id,
                                                         max_char_length, 
                                                         max_word_length,
                                                         args.gpu))

    test_loader = DataLoader(DidData(test_insts), 
                             batch_size=args.batch_size, 
                             shuffle=False,
                             collate_fn=functools.partial(collate, 
                                                          char_to_id, 
                                                          word_to_id,
                                                          label_to_id,
                                                          dist_sizes,
                                                          gold_to_id,
                                                          max_char_length, 
                                                          max_word_length,
                                                          args.gpu))

    # golds = []
    # guesses = []
    # for _, _, _, ss, l in dev_insts:
    #     #print(sum([x[0] for x in ss[0]]))
    #     golds.append(l)
    #     guesses.append(sorted(ss[0], reverse=True)[0][1])
    # print(f1_score(golds, guesses, average="macro"))
    # print(accuracy_score(golds, guesses))
    # sys.exit()

    model = Ensemble(gold_to_id,
                     label_to_id,
                     dist_sizes,
                     [char_to_id, word_to_id],
                     [max_char_length, max_word_length],
                     [args.character_embeddings, args.word_embeddings],
                     [args.freeze_character_embeddings, args.freeze_word_embeddings],
                     args.dropout,
                     
                     # CNN
                     [args.filter_count, args.filter_count],
                     [list(map(int, [] if args.char_kernel_sizes == None else args.char_kernel_sizes.split(","))), list(map(int, [] if args.word_kernel_sizes == None else args.word_kernel_sizes.split(",")))],
                     #[args.dropout, args.dropout],
                     #[512, 512],

                     # RNN
                     [args.char_rnn_hidden_size, args.word_rnn_hidden_size],
                     [args.rnn_dropout, args.rnn_dropout],

                     # MLP
                     #[26],
                     #[[]],
                     #[len(tlang_to_id), 26],
                     #[[128, 32], [128, 32]],
            )

    # if GPU is to be used
    if args.gpu:
        model.cuda()

    print(model)

    model.apply(init_weights)
    metric = torch.nn.KLDivLoss(reduction="batchmean")
    for a, name in [(0, "cnn_submodels"), (0, "rnn_submodels"), (1, "dist_submodels"), (2, "label_submodels")]:
        for b in range(len(getattr(model, name))):
            optim = SGD(getattr(model, name)[b].parameters(), lr=args.learning_rate, momentum=args.momentum)
            sched = ReduceLROnPlateau(optim, patience=args.patience, verbose=True)
            best_dev_loss = None
            since_improvement = 0
            for epoch in range(1, args.pretrain_epochs):

                loss_total = 0.0
                item_total = 0
                losses = []
                for i, tpl in enumerate(train_loader):
                    model.zero_grad()
                    x = tpl[a][b]
                    y = tpl[3]
                    out = getattr(model, name)[b](x) #F.log_softmax(m(x), dim=1)
                    loss = metric(out, y)
                    losses.append(float(loss))
                    loss = torch.mean(loss)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    loss_total += loss * y.shape[0]
                    item_total += y.shape[0]
                dev_loss, dev_f1, dev_acc, dev_out = evaluate_model(dev_loader, model, metric, name, a, b)
                logging.info("Epoch %.4d: Train loss = %.3f\tDev loss/f1/accuracy = %.3f/%.3f/%.3f", epoch, loss_total / item_total, dev_loss, dev_f1, dev_acc)
                if epoch > 0:
                    sched.step(dev_loss)
                    if best_dev_loss == None or dev_loss < best_dev_loss:
                        since_improvement = 0
                        best_dev_loss = dev_loss
                        best_dev_out = dev_out
                        #best_test_out = apply_model(dev_loader, model, metric)
                    else:
                        since_improvement += 1
                        if since_improvement > args.early_stop:
                            logging.info("Stopping early after %d epochs with no improvement", args.early_stop)
                            break


    for a, name in [(0, "cnn_submodels"), (0, "rnn_submodels"), (1, "dist_submodels"), (2, "label_submodels")]:
        for b in range(len(getattr(model, name))):
            dev_loss, dev_f1, dev_acc, dev_out = evaluate_model(dev_loader, model, metric, name, a, b)
            print(getattr(model, name)[b])
            print(dev_loss, dev_f1, dev_acc)
            if args.freeze_submodels:
                for param in getattr(model, name)[b].parameters():
                    param.requires_grad = False


    #
    # Full model
    #
    optim = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    sched = ReduceLROnPlateau(optim, patience=args.patience, verbose=True)
    best_dev_loss = None
    best_dev_f1 = None
    since_improvement = 0
    for epoch in range(1, args.epochs):

        loss_total = 0.0
        item_total = 0
        losses = []
        for i, (seq, dists, labelings, y) in enumerate(train_loader):
            model.zero_grad()
            out = model(seq, dists, labelings)
            loss = metric(out, y)
            losses.append(float(loss))
            loss = torch.mean(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_total += loss * y.shape[0]
            item_total += y.shape[0]
        dev_loss, dev_f1, dev_acc, dev_out = evaluate_model(dev_loader, model, metric)
        logging.info("Epoch %.4d: Train loss = %.3f\tDev loss/f1/accuracy = %.3f/%.3f/%.3f", epoch, loss_total / item_total, dev_loss, dev_f1, dev_acc)
        if epoch > 0:
            #sched.step(dev_loss)
            sched.step(-dev_f1)
            #if best_dev_loss == None or dev_loss < best_dev_loss:
            if best_dev_f1 == None or dev_f1 > best_dev_f1:
                since_improvement = 0
                best_dev_f1 = dev_f1
                best_dev_out = dev_out
                best_test_out = apply_model(test_loader, model, metric)
            else:
                since_improvement += 1
                if since_improvement > args.early_stop:
                    logging.info("Stopping early after %d epochs with no improvement", args.early_stop)
                    break

    print(best_dev_f1)
    for a, name in [(0, "cnn_submodels"), (0, "rnn_submodels"), (1, "dist_submodels"), (2, "label_submodels")]:
        for b in range(len(getattr(model, name))):
            #ram in getattr(model, name)[b].get_param
            dev_loss, dev_f1, dev_acc, dev_out = evaluate_model(dev_loader, model, metric, name, a, b)
            print(getattr(model, name)[b])
            print(dev_loss, dev_f1, dev_acc)
                
    id_to_gold = {v : k for k, v in gold_to_id.items()}
    with open(args.dev_output, "wt") as ofd:
        for item, gold, guess in best_dev_out:
            ofd.write("{}\t{}\t{}\t{}\n".format(id_to_gold[gold], id_to_gold[guess], item, "N/A"))

    with open(args.test_output, "wt") as ofd:
        for item, guess in best_test_out:
            ofd.write("{}\t{}\t{}\t{}\n".format("N/A", id_to_gold[guess], item, "N/A"))
