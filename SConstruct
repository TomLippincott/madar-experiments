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
    ("MADAR_PATH", "", "/export/common/data/corpora/MADAR/clean"),
    ("GRID_RESOURCES", "", ["mem_free=32G"]),
)

main_tasks = ["madar_task_1_26", "madar_task_2"]
#main_tasks = ["madar_task_2"]

secondary_tasks = [] #"madar_task_1_6", "lid", "ccb"]

learning_rates = [0.1]
momentums = [0.9] #, 0.5, 0.0] #[0:1]
dropouts = [0.5]
batch_sizes = [512] #[8, 32, 128, 512, 1024]
filter_counts = [100] #, 200, 300]
patiences = [10]
early_stops = [20]
char_kernel_sizess = [
    "1,2,3,4,5",
]
word_kernel_sizess = [
    "1,2,3,4,5",
]

lm_sizes = [1, 2, 3, 4, 5]

word_embeddings = {
    "madar" : "/exp/pshapiro/arabic_dialects_project/shared_task/fastText.bin",
    "madar_ext" : "/exp/pshapiro/arabic_dialects_project/shared_task/fastText.extra_data.300.bin",
    "wikipedia_cc" : "data/cc.ar.300.bin",
    #"100" : 100,
    "300" : 300,
}

character_embeddings = {
    "100" : 100,
    #"300" : 300,
}

ns = {
    "1" : 1,
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5" : 5,
}

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  BUILDERS={"ExtractTwitter" : Builder(action="python3 scripts/extract_twitter.py -i ${SOURCES[0]} -o ${TARGETS[0]}"),
                            "ExtractReddit" : Builder(action="python3 scripts/extract_reddit.py -i ${SOURCES[0]} -o ${TARGETS[0]}"),
                            "Cat" : Builder(action="cat ${SOURCES[0]}|perl -pe '$$_=~s/^(\S+)(\s+)(\S.*)$$/\\3\\t\\1/;' > ${TARGETS[0]}"),
                            "Aggregate" : Builder(action="python3 scripts/aggregate_users.py -i ${SOURCES[0]} -o ${TARGETS[0]}"),
                            "Unaggregate" : Builder(action="python3 scripts/unaggregate_users.py -i ${SOURCES[0]} -s ${SOURCES[1]} -r ${SOURCES[2]} -o ${TARGETS[0]}"),
                            "TrainValid" : Builder(action="python3 scripts/train_valid.py --train ${SOURCES[0].abspath} --n ${N} --model ${TARGETS[0].abspath} ${'--normalize' if NORMALIZE else ''} ${'--add_user' if ADD_USER else ''}"),
                            "ApplyValid" : Builder(action="python3 scripts/apply_valid.py --model ${SOURCES[0].abspath} --input ${SOURCES[1].abspath} --n ${N} --output ${TARGETS[0].abspath} ${'--normalize' if NORMALIZE else ''} ${'--add_user' if ADD_USER else ''}"),
                            "TrainEnsemble" : Builder(action=["module load cuda90/toolkit", "python3 scripts/train_ensemble.py --train ${SOURCES[0]} --dev ${SOURCES[1]} --test ${SOURCES[2]} --gpu --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --patience ${PATIENCE} --word_embeddings ${WORD_EMBEDDINGS} --character_embeddings ${CHARACTER_EMBEDDINGS} --seed ${SEED} --char_kernel_sizes ${CHAR_KERNEL_SIZES} --word_kernel_sizes ${WORD_KERNEL_SIZES} ${' --freeze_word_embeddings' if FREEZE_WORD_EMBEDDINGS else ''} ${' --freeze_character_embeddings' if FREEZE_CHARACTER_EMBEDDINGS else ''} --filter_count ${FILTER_COUNT} --dev_output ${TARGETS[0]} --test_output ${TARGETS[1]} --epochs 1000 ${'--freeze_submodels' if FREEZE_SUBMODELS else ''} --char_rnn_hidden_size 128 --word_rnn_hidden_size 128 ${SOURCES[3:]}"]),
                            "Evaluate" : Builder(action="python3 scripts/evaluate.py --input ${SOURCES[0]} --output ${TARGETS[0]}"),
                            "CollateResults" : Builder(action="python3 scripts/collate_results.py --output ${TARGETS[0]} ${SOURCES}"),
                            "PlotConfusion" : Builder(action="python3 scripts/plot_confusion.py --input ${SOURCES[0]} --output ${TARGETS[0]}"),
                            "MajorityVote" : Builder(action="python3 scripts/majority_vote.py --input ${SOURCES[0]} --reference ${SOURCES[1]} --full ${SOURCES[2]} --output ${TARGETS[0]}"),
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

# collect experimental outputs here for evaluation
to_evaluate = []


#
# Language models
#
language_model_scores = {}
for n in lm_sizes:
    for task in main_tasks + secondary_tasks:
        for add_user in ([True, False] if task == "madar_task_2" else [False]):
            for normalize in ([True, False] if task == "madar_task_2" else [False]):
                model = env.TrainValid(
                    "work/language_models/${TASK}/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.model.txt",
                    "${MADAR_PATH}/${TASK}_train.txt.gz",
                    N=n,
                    TASK=task,
                    ADD_USER=add_user,
                    NORMALIZE=normalize,
                )
                for other_task in main_tasks + secondary_tasks:
                    if other_task != task or task not in main_tasks:
                        continue
                    train_scored = env.ApplyValid(
                        "work/applied_language_models/${OTHER_TASK}/${TASK}/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.train.scores.txt",
                        [model, "${MADAR_PATH}/${OTHER_TASK}_train.txt.gz"],
                        N=n,
                        TASK=task,
                        OTHER_TASK=other_task,
                        ADD_USER=add_user,
                        NORMALIZE=normalize,
                    )
                    dev_scored = env.ApplyValid(
                        "work/applied_language_models/${OTHER_TASK}/${TASK}/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.dev.scores.txt",
                        [model, "${MADAR_PATH}/${OTHER_TASK}_dev.txt.gz"],
                        N=n,
                        TASK=task,
                        OTHER_TASK=other_task,
                        ADD_USER=add_user,
                        NORMALIZE=normalize,
                    )
                    test_scored = env.ApplyValid(
                        "work/applied_language_models/${OTHER_TASK}/${TASK}/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.test.scores.txt",
                        [model, "${MADAR_PATH}/${OTHER_TASK}_test.txt.gz"],
                        N=n,
                        TASK=task,
                        OTHER_TASK=other_task,
                        ADD_USER=add_user,
                        NORMALIZE=normalize,
                    )
                    key = (other_task, n, add_user, normalize)
                    language_model_scores[key] = language_model_scores.get(key, {})
                    language_model_scores[key][task] = [train_scored, dev_scored, test_scored]
                    if other_task == task and task in main_tasks:
                        to_evaluate += [
                            #{"model_type" : "lm", "split_type" : "train", "task_name" : task, "scores" : train_scored, "n" : n, "normalize" : normalize, "add_user" : add_user},
                            {"model_type" : "lm", "split_type" : "dev", "task_name" : task, "scores" : dev_scored, "n" : n, "normalize" : normalize, "add_user" : add_user},
                            #{"model_type" : "lm", "split_type" : "test", "task_name" : task, "scores" : test_scored, "n" : n, "normalize" : normalize, "add_user" : add_user},
                        ]


#
# User-aggregated experiments
#
agg_train, agg_dev, agg_test = [env.Aggregate("work/aggregated/${SPLIT}.txt.gz", 
                                              "${MADAR_PATH}/${TASK}_${SPLIT}.txt.gz",
                                              TASK="madar_task_2",
                                              SPLIT=s) for s in ["train", "dev", "test"]]
for n in lm_sizes:
    for add_user in [True, False]:
        for normalize in [True, False]:
            model = env.TrainValid(
                "work/aggregated/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.model.txt",
                agg_train,
                N=n,
                ADD_USER=add_user,
                NORMALIZE=normalize,
            )
            agg_train_scored = env.ApplyValid(
                "work/aggregated/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.train.aggscores.txt",
                [model, agg_train],
                N=n,
                ADD_USER=add_user,
                NORMALIZE=normalize,
            )

            agg_dev_scored = env.ApplyValid(
                "work/aggregated/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.dev.aggscores.txt",
                [model, agg_dev],
                N=n,
                ADD_USER=add_user,
                NORMALIZE=normalize,
            )

            agg_test_scored = env.ApplyValid(
                "work/aggregated/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.test.aggscores.txt",
                [model, agg_test],
                N=n,
                ADD_USER=add_user,
                NORMALIZE=normalize,
            )

            # dev_scored = env.Unaggregate(
            #     "work/aggregated/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.dev.scores.txt",
            #     [agg_dev, agg_dev_scored, "${MADAR_PATH}/${TASK}_dev.txt.gz"],
            #     TASK="madar_task_2",
            #     N=n,
            #     ADD_USER=add_user,
            #     NORMALIZE=normalize,
            # )

            # test_scored = env.Unaggregate(
            #     "work/aggregated/ppm-char-${N}-${NORMALIZE}-${ADD_USER}.test.scores.txt",
            #     [agg_test, agg_test_scored, "${MADAR_PATH}/${TASK}_test.txt.gz"],
            #     TASK="madar_task_2",
            #     N=n,
            #     ADD_USER=add_user,
            #     NORMALIZE=normalize,
            # )

            to_evaluate.append({"model_type" : "agg_lm", "split_type" : "dev", "task_name" : "madar_task_2", "scores" : agg_dev_scored, "n" : n, "normalize" : normalize, "add_user" : add_user})





#
# Main experiments
#
fold = 1
for task in main_tasks:
    continue
    n, normalize, add_user = (4, False, True) if task == "madar_task_2" else (4, False, False)
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
                                                config = "task_name={}-word_emb={}-char_emb={}-lr={}-momentum={}-dropout={}-bs={}-fc={}-pat={}-es={}-cks={}-wks={}".format(task, wn, cn, learning_rate, momentum, dropout, batch_size, filter_count, patience, early_stop, cks, wks).replace(",", "_")

                                                key = (task, n, add_user, normalize)
                                                lm_scores = sum([v for k, v in language_model_scores[key].items()], [])
                                                # if task == "madar_task_2":
                                                #     key = (task, n, False, False)
                                                #     lm_scores += sum([v for k, v in language_model_scores[key].items() if k != "madar_task_2"], [])
                                                #     dev_scored, test_scored = map(env.File, ["work/experiments/{}.dev.out".format(config),
                                                #                                "work/experiments/{}.test.out".format(config)])
                                                dev_scored, test_scored = env.TrainEnsemble(
                                                    ["work/experiments/{}.dev.out".format(config),
                                                     "work/experiments/{}.test.out".format(config)],
                                                    ["${MADAR_PATH}/${TASK}_train.txt.gz",
                                                     "${MADAR_PATH}/${TASK}_dev.txt.gz",
                                                     "${MADAR_PATH}/${TASK}_test.txt.gz"] + lm_scores,
                                                    TASK=task,
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
                                                    FREEZE_WORD_EMBEDDINGS=False,
                                                    SEED=fold,
                                                    CHAR_KERNEL_SIZES=cks,
                                                    WORD_KERNEL_SIZES=wks,
                                                    FREEZE_SUBMODELS=True,
                                                )
                                                to_evaluate.append(
                                                    {"model_type" : "ensemble",
                                                     "split_type" : "dev",
                                                     "task_name" : task,
                                                     "scores" : dev_scored,
                                                     "word_embeddings" : wn,
                                                     #"normalize" : False,
                                                     #"add_user" : False,
                                                 }
                                                )



#
# Evaluation
#
evals = []
for experiment in to_evaluate:
    scores_file = experiment["scores"]
    task_name = experiment["task_name"]
    #norm = experiment.get("normalize", False)
    #add_user = experiment.get("add_user", False)
    #if task_name not in main_tasks or experiment["split_type"] != "test":
    config = "-".join(sorted(["{}={}".format(k, v) for k, v in experiment.items() if k not in ["task_name", "scores"]]))
    if "agg" not in experiment["model_type"] and task_name == "madar_task_2":
        scores_file = env.MajorityVote("work/evaluations/${TASK_NAME}/%s_majority.txt" % config, [scores_file, "/export/common/data/corpora/MADAR/MADAR-Shared-Task-Subtask-2/MADAR-Twitter-Subtask-2.DEV.user-label.tsv", "/export/common/data/corpora/MADAR/MADAR-Shared-Task-Subtask-2/MADAR-Twitter-Subtask-2.dev.user-tweets-features.tsv.hyd"], TASK_NAME=task_name)
        #continue
    ev = env.Evaluate("work/evaluations/${TASK_NAME}/%s.pkl.gz" % config, scores_file, TASK_NAME=task_name)
    env.PlotConfusion("work/heatmaps/${TASK_NAME}/%s.png" % config, ev, TASK_NAME=task_name)
    evals.append(ev)

if len(evals) > 0:
    col = env.CollateResults("work/collated.txt", evals)
