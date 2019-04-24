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

#            submodels.append(CNN(f2i, ml, emb, fr, fc, kw, do))
class CNN(torch.nn.Module):
    def __init__(self,
                 feat_to_id,
                 max_length,
                 embeddings,
                 freeze_embeddings,
                 filter_count, 
                 kernel_widths, 
                 dropout_prob, 
                 output_size,
             ):
        super(CNN, self).__init__()

        # set a bunch of dimensions and such
        self._filter_count = filter_count
        self._kernel_widths = [c for c in kernel_widths]
        self._max_length = max_char_length
        self._nfeats = len(feat_to_id)
        self._dropout_prob = dropout_prob
        self._output_size = output_size
        
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

        # dropout layer of specified probability
        self.dropout = torch.nn.Dropout(self._dropout_prob)

        # final fully-connected linear layer over the outputs from all the convolutions
        self.output = torch.nn.Linear(len(self._kernel_widths) * self._filter_count, 
                                      self._output_size)

    @property
    def out_size(self):
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
        x = self.dropout(x)
        return self.output(x)


class RNN(torch.nn.Module):
    def __init__(self, 
                 feat_to_id,
                 max_length, 
                 embeddings,
                 freeze_embeddings,
                 dropout_prob,
                 output_size,
             ):
        super(RNN, self).__init__()

        # set a bunch of dimensions and such
        self._max_length = max_char_length
        self._nfeats = len(feat_to_id)
        self._dropout_prob = dropout_prob
        self._hidden_size = output_size
        
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

        self.rnn = torch.nn.GRU(input_size=self._embedding_size,
                                     hidden_size=self._hidden_size,
                                     num_layers=1,
                                     batch_first=True,
                                     dropout=self._dropout_prob,
                                     bidirectional=False)

        # dropout layer of specified probability
        self.dropout = torch.nn.Dropout(self._dropout_prob)

    @property
    def out_size(self):
        return self._hidden_size
        
    def forward(self, x):
        # go from sequences of characters/words to corresponding sequences of embeddings
        x = self._embeddings(x)
        out, h = self.rnn(x)
        return self.dropout(h.squeeze(0))


class MLP(torch.nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size_list,
                 dropout_prob, 
             ):
        super(MLP, self).__init__()

        # set a bunch of dimensions and such
        self._size_list = [input_size] + hidden_size_list
        self._dropout_prob = dropout_prob
        
        # dropout layer of specified probability
        self.dropout = torch.nn.Dropout(self._dropout_prob)
        self.layers = torch.nn.ModuleList()
        for i in range(len(self._size_list) - 1):
            self.layers.append(torch.nn.Linear(self._size_list[i], self._size_list[i + 1])) 
            
    def forward(self, x):
        for i in range(len(self._size_list) - 1):
            x = F.relu(self.layers[i](x))
        return x

    @property
    def out_size(self):
        return self._size_list[-1]

    
class Ensemble(torch.nn.Module):
    def __init__(self,
                 label_to_id,
                 feats_to_ids,
                 max_lengths,
                 embeddings,
                 freeze,
                 
                 filter_counts, 
                 kernel_widths, 
                 cnn_dropout_probs,
                 cnn_output_sizes,

                 rnn_hidden_sizes,
                 rnn_dropout_probs,

                 mlp_input_sizes,
                 mlp_hidden_size_lists,
                 mlp_dropout_probs,
             ):
        super(Ensemble, self).__init__()
        self._output_size = len(label_to_id)
        self.cnn_submodels = torch.nn.ModuleList()
        for f2i, ml, emb, fr, fc, kw, do, out in zip(feats_to_ids, max_lengths, embeddings, freeze, filter_counts, kernel_widths, cnn_dropout_probs, cnn_output_sizes):
            self.cnn_submodels.append(CNN(f2i, ml, emb, fr, fc, kw, do, out))
        self.rnn_submodels = torch.nn.ModuleList()
        for f2i, ml, emb, fr, do, hidden in zip(feats_to_ids, max_lengths, embeddings, freeze, rnn_dropout_probs, rnn_hidden_sizes):
            self.rnn_submodels.append(RNN(f2i, ml, emb, fr, do, hidden))
        self.mlp_submodels = torch.nn.ModuleList()
        for i, hs, do in zip(mlp_input_sizes, mlp_hidden_size_lists, mlp_dropout_probs):
            self.mlp_submodels.append(MLP(i, hs, do))
        self._combined_size = sum([x.out_size for x in self.cnn_submodels] + [x.out_size for x in self.rnn_submodels] + [x.out_size for x in self.mlp_submodels])
        print(self._combined_size)
        self.linear = torch.nn.Linear(self._combined_size, self._output_size)

        
    def forward(self, sequence_data, mlp_data):
        outputs = []
        for submodel, data in zip(self.cnn_submodels, sequence_data):
            outputs.append(submodel(data))
        for submodel, data in zip(self.rnn_submodels, sequence_data):
            outputs.append(submodel(data))
        for submodel, data in zip(self.mlp_submodels, mlp_data):
            outputs.append(submodel(data))
        inp = torch.cat(outputs, 1)
        return self.linear(inp)
    
        
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
def collate(clu, wlu, llu, tlu, max_char_length, max_word_length, gpu, tuples):
    y = numpy.zeros(shape=(len(tuples), len(llu)))
    tl = numpy.zeros(shape=(len(tuples), len(tlu)))
    sc = numpy.zeros(shape=(len(tuples), 26))
    for i, (_, t, s, _, _, l) in enumerate(tuples):
        y[i, llu[l]] = 1.0
        tl[i, tlu[t]] = 1.0
        sc[i, :] = s
    wx = numpy.zeros(shape=(len(tuples), max_word_length))
    cx = numpy.zeros(shape=(len(tuples), max_char_length))
    for i, (_, _, _, _chars, _words, _) in enumerate(tuples):
        chars = [clu.get(c, 1) for c in _chars][0:max_char_length]
        cx[i, 0:len(chars)] = chars
        words = [wlu.get(c, 1) for c in _words][0:max_word_length]
        wx[i, 0:len(words)] = words
    
    cx, wx, tl, sc, y = [(x.cuda() if gpu else x) for x in [torch.LongTensor(cx), torch.LongTensor(wx), torch.FloatTensor(tl), torch.FloatTensor(sc), torch.Tensor(y)]]
    return ([cx, wx], [tl, sc], y)
    
        
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
    for i, (seq, mlp, y) in enumerate(loader, 1):
        out = model(seq, mlp)
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
            char_seq = char_seq if not args.max_char_length else char_seq[:args.max_char_length]
            word_seq = word_seq if not args.max_word_length else word_seq[:args.max_word_length]
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            train.append((None, None, None, char_seq, word_seq, label))
        for line in ifd:
            try:
                uid, _, tw, feats, label, char_seq = line.strip().split("\t")
                feats = [float(x) for x in feats.split(",")]
            except:
                uid, tw, feats = (None, None, None)
                char_seq, label = line.strip().split("\t")
            if args.normalize:
                char_seq = normalize_doc(char_seq)
            if char_seq == None:
                continue
            word_seq = char_seq.split()
            char_seq = char_seq if not args.max_char_length else char_seq[:args.max_char_length]
            word_seq = word_seq if not args.max_word_length else word_seq[:args.max_word_length]
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            train.append((uid, tw, feats, char_seq, word_seq, label))

    random.shuffle(train)
    train = train[0:len(train) if args.train_count == None else args.train_count]


    label_to_id = {} #"<UNK>" : 0}
    char_to_id = {"<PAD>" : 0, "<UNK>" : 1}
    word_to_id = {"<PAD>" : 0, "<UNK>" : 1}
    tlang_to_id = {"<UNK>" : 0}
    for (_, tw, _, cs, ws, l) in train:
        for c in cs:
            char_to_id[c] = char_to_id.get(c, len(char_to_id))
        for w in ws:
            word_to_id[w] = word_to_id.get(w, len(word_to_id))
        label_to_id[l] = label_to_id.get(l, len(label_to_id))
        tlang_to_id[tw] = tlang_to_id.get(tw, len(tlang_to_id))
    
    with open(args.dev, "rt") as ifd:
        line = ifd.readline()
        try:
            _, _, _, _, _, _ = line.strip().split("\t")
        except:
            char_seq, label = line.strip().split("\t")
            word_seq = char_seq.split()
            char_seq = char_seq if not args.max_char_length else char_seq[:args.max_char_length]
            word_seq = word_seq if not args.max_word_length else word_seq[:args.max_word_length]
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            dev.append((None, None, None, char_seq, word_seq, label))
        for line in ifd:
            try:
                uid, _, tw, feats, label, char_seq = line.strip().split("\t")
                feats = [float(x) for x in feats.split(",")]
            except:
                char_seq, label = line.strip().split("\t")
                uid, tw, feats = (None, None, None)
            if args.normalize:
                char_seq = normalize_doc(char_seq)
            if char_seq == None:
                continue
            word_seq = char_seq.split()
            char_seq = char_seq if not args.max_char_length else char_seq[:args.max_char_length]
            word_seq = word_seq if not args.max_word_length else word_seq[:args.max_word_length]
            max_char_length = max(max_char_length, len(char_seq))
            max_word_length = max(max_word_length, len(word_seq))
            dev.append((uid, tw, feats, char_seq, word_seq, label))

    dev = dev[0:len(dev) if args.dev_count == None else args.dev_count]

    for (_, _, _, cs, ws, l) in dev:
        for c in cs:
            char_to_id[c] = char_to_id.get(c, len(char_to_id))
        for w in ws:
            word_to_id[w] = word_to_id.get(w, len(word_to_id))
        label_to_id[l] = label_to_id.get(l, len(label_to_id))
    id_to_label = {v : k for k, v in label_to_id.items()}
    id_to_tlang = {v : k for k, v in tlang_to_id.items()}

    logging.info("Loaded %d train and %d dev instances, with %d features and %d labels, maximum character sequence length %d, maximum word sequence length %d", len(train), len(dev), len(char_to_id), len(label_to_id), max_char_length, max_word_length)
    
    train_loader = DataLoader(DidData(train), 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=functools.partial(collate, 
                                                           char_to_id, 
                                                           word_to_id,
                                                           label_to_id,
                                                           tlang_to_id,
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
                                                         tlang_to_id,
                                                         max_char_length, 
                                                         max_word_length,
                                                         args.gpu))

    model = Ensemble(label_to_id,
                     [char_to_id, word_to_id],
                     [max_char_length, max_word_length],
                     [args.character_embeddings, args.word_embeddings],
                     [args.freeze_character_embeddings, args.freeze_word_embeddings],
                     
                     # CNN
                     [args.filter_count, args.filter_count],
                     [list(map(int, args.char_kernel_sizes.split(","))), list(map(int, args.word_kernel_sizes.split(",")))],
                     [args.dropout, args.dropout],
                     [512, 512],

                     # RNN
                     [64, 64],
                     [args.dropout, args.dropout],

                     # MLP
                     [len(tlang_to_id), 26],
                     [[128, 32], [128, 32]],
                     [args.dropout, args.dropout],
            )
    #print(model)
    
    # if GPU is to be used
    if args.gpu:
        model.cuda()

    # recursively initialize model weights
    model.apply(init_weights)
    metric = torch.nn.KLDivLoss(reduction="batchmean")
    optim = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    sched = ReduceLROnPlateau(optim, patience=args.patience, verbose=True)
    #sys.exit()
    best_dev_loss = None
    best_dev_out = None
    since_improvement = 0
    for epoch in range(1, args.epochs):
        loss_total = 0.0
        for i, (seq, mlp, y) in enumerate(train_loader):
            model.zero_grad()            
            out = model(seq, mlp)
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
