import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from glob import glob
from steamroller.scons import GridBuilder as Builder


# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 100),
    ("TWITTER_PATH", "", "/twitter/current"),
    ("REDDIT_PATH", "", "/export/common/data/corpora/reddit"),
    ("CCB_PATH", "", "/expscratch/tlippincott/ccb_dialect"),
    ("MADAR_PATH", "", "/expscratch/tlippincott/data/MADAR"),
    ("LEARNING_RATE", "", 0.1),
    ("MOMENTUM", "", 0.9),
    ("DROPOUT", "", 0.5),
    ("PATIENCE", "", 10),
    ("EARLY_STOP", "", 20),
    ("WORD_EMBEDDINGS", "", None),
    ("CHARACTER_EMBEDDINGS", "", None),
    ("FREEZE_WORD_EMBEDDINGS", "", False),
    ("FREEZE_CHARACTER_EMBEDDINGS", "", False),
    ("BATCH_SIZE", "", 16),
    ("FILTER_COUNT", "", 100),
    ("CHAR_KERNEL_SIZES", "", "1,2,3,4,5"),
    ("WORD_KERNEL_SIZES", "", "1,2,3,4,5"),
    ("FOLDS", "", 5),
)

tasks = {
#    "madar_1_6" : "/home/hltcoe/tlippincott/data/MADAR/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-{}.tsv",
#    "madar_1_26" : "/home/hltcoe/tlippincott/data/MADAR/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-26-{}.tsv",
    "madar_2" : "/home/hltcoe/tlippincott/data/MADAR/MADAR-Shared-Task-Subtask-2/MADAR-Twitter-Subtask-2.{}.user-tweets-features.tsv.hyd",
    #"lid" : "data/lid_{}.txt",
}

learning_rates = [0.1]
momentums = [0.9], #, 0.5, 0.0] #[0:1]
dropouts = [0.5]
batch_sizes = [512] #[8, 32, 128, 512, 1024]
filter_counts = [100, 200, 300]
patiences = [10]
early_stops = [20]
char_kernel_sizess = [
    "0",
    #"1",
    #"1,2,3",
    "1,2,3,4,5",
]
word_kernel_sizess = [
    "0",
    #"1",
    #"1,2,3",
    "1,2,3,4,5",
]

word_embeddings = {
    "madar" : "/exp/pshapiro/arabic_dialects_project/shared_task/fastText.bin",
    "wikipedia_cc" : "data/cc.ar.300.bin",
    "100" : 100,
    #"300" : 300,
}

character_embeddings = {
    "100" : 100,
    #"300" : 300,
}

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  BUILDERS={"ExtractTwitter" : Builder(action="python3 scripts/extract_twitter.py -i ${SOURCES[0]} -o ${TARGETS[0]}"),
                            "ExtractReddit" : Builder(action="python3 scripts/extract_reddit.py -i ${SOURCES[0]} -o ${TARGETS[0]}"),
                            "CatZip" : Builder(action="cat ${SOURCES[0]}|gzip > ${TARGETS[0]}"),
                            "TrainCNN" : Builder(action=["module load cuda90/toolkit", "python3 scripts/train_cnn.py --train ${SOURCES[0]} --dev ${SOURCES[1]} --gpu --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --patience ${PATIENCE} --word_embeddings ${WORD_EMBEDDINGS} --character_embeddings ${CHARACTER_EMBEDDINGS} --seed ${SEED} --output ${TARGETS[0]} --char_kernel_sizes ${CHAR_KERNEL_SIZES} --word_kernel_sizes ${WORD_KERNEL_SIZES} ${' --freeze_word_embeddings' if FREEZE_WORD_EMBEDDINGS else ''} ${' --freeze_character_embeddings' if FREEZE_CHARACTER_EMBEDDINGS else ''} --filter_count ${FILTER_COUNT}"]),

                  },
                  tools=["default"],
)



# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")

# reddit_comments = env.Glob("${REDDIT_PATH}/comments/*")
# reddit_submissions = env.Glob("${REDDIT_PATH}/submissions/*")

# twitter_ame = env.Glob("${TWITTER_PATH}/africa_middle_east/*/*")
# twitter_sample = env.Glob("${TWITTER_PATH}/sample/*/*")

# for fname in (reddit_comments + reddit_submissions):
#     base = os.path.splitext(os.path.basename(fname.rstr()))[0]
#     target = os.path.join("work", "reddit", "{}.json.gz".format(base))
#     env.ExtractReddit(target, fname)

# for name, data in [("africa_middle_east", twitter_ame), ("sample", twitter_sample)]:
#     for dname in data:
#         rest, month = os.path.split(dname.rstr())
#         _, year = os.path.split(rest)
#         target = os.path.join("work", "twitter", name, "{}_{}.json.gz".format(year, month))
#         env.ExtractTwitter(target, dname)

# env.CatZip("work/ccb_dialect.txt.gz", "${CCB_PATH}/multiclass.labeled.txt")

fold = 1
for task_name, task_pattern in tasks.items():
    train = task_pattern.format("train")
    dev = task_pattern.format("dev")
    for wn, we in word_embeddings.items():
        for cn, ce in character_embeddings.items():
            for learning_rate in learning_rates:
                for momentum in momentums:
                    for dropout in dropouts:
                        for batch_size in batch_sizes:
                            for filter_count in filter_counts:
                                for patience in patiences:
                                    for early_stop in early_stops:
                                        for cks in char_kernel_sizess:
                                            for wks in word_kernel_sizess:
                                                if cks == wks and wks == "0":
                                                    continue
                                                env.TrainCNN("work/experiments/${TASK_NAME}/cnn-${LEARNING_RATE}-${MOMENTUM}-${DROPOUT}-${BATCH_SIZE}-${FILTER_COUNT}-${PATIENCE}-${EARLY_STOP}-${CHARACTER_EMBEDDINGS_NAME}-${WORD_EMBEDDINGS_NAME}-${CHAR_KERNEL_SIZES.replace(',', '_')}-${WORD_KERNEL_SIZES.replace(',', '_')}.out", 
                                                             [train, dev],
                                                             TASK_NAME=task_name,
                                                             GRID_RESOURCES=["h_rt=24:0:0", "gpu=1"],
                                                             GRID_QUEUE="gpu.q@@1080",
                                                             LEARNING_RATE=learning_rate,
                                                             MOMENTUM=momentum,
                                                             DROPOUT=dropout,
                                                             BATCH_SIZE=batch_size,
                                                             FILTER_COUNT=filter_count,
                                                             PATIENCE=patience,
                                                             EARLY_STOP=early_stop,
                                                             WORD_EMBEDDINGS=we,
                                                             CHARACTER_EMBEDDINGS=ce,
                                                             WORD_EMBEDDINGS_NAME=wn,
                                                             CHARACTER_EMBEDDINGS_NAME=cn,
                                                             SEED=fold,
                                                             CHAR_KERNEL_SIZES=cks,
                                                             WORD_KERNEL_SIZES=wks,
                                                         )

