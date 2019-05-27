# MADAR Arabic dialect experiments

**Please ping me (tom@cs.jhu.edu) if you would like to use this repository beyond it's current archival function, and I can clean it up for you!**

Install dependencies:

```
pip install -r requirements.txt --no-cache-dir --user
```

A reasonable command:

```
python3 scripts/train_cnn.py --train $TRAIN --dev $DEV --learning_rate 0.1 --gpu --batch_size 512 --patience 10 --early_stop 20 --word_embeddings $EMBS --character_embeddings 100 --char_kernel_sizes 1,2,3 --word_kernel_sizes 1,2,3 --filter_count 100 --normalize --output out.txt
```

Where your data might be:

```
TRAIN=/home/hltcoe/tlippincott/data/MADAR/MADAR-Shared-Task-Subtask-2/MADAR-Twitter-Subtask-2.train.user-tweets-features.tsv.hyd
DEV=/home/hltcoe/tlippincott/data/MADAR/MADAR-Shared-Task-Subtask-2/MADAR-Twitter-Subtask-2.train.user-tweets-features.tsv.hyd
```

and word embeddings might be:

```
EMBS=/exp/pshapiro/arabic_dialects_project/shared_task/fastText.bin
```

or

```
EMBS=/expscratch/tlippincott/arabic-lid-data/data/cc.ar.300.bin
```

Note that word and character embeddings can either be a file (for pretrained) or an integer (for randomly-initialized).  Data files can be in one of the MADAR formats, or simply each line "TEXT<TAB>LABEL".  The "--normalize" switch removes all non-Arabic-or-punctuation/whitespace characters and make a few substitutions.  The output file will contain the dev instances as "GUESS<TAB>GOLD<TAB>TEXT" for later scrutiny, while the scores during training are written to stdout.  You should probably run on a GPU w/ minibatches of 512, but you can also remove the "--gpu" switch if desired.
