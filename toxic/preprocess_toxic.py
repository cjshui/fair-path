import numpy as np
import pandas as pd
from sklearn.utils import axis0_safe_slice
import h5py
import torch

import os.path as osp
import os
from tqdm import tqdm

from transformers import pipeline


def clean_toxic_raw(df_raw, split="train"):
    ### See data descriptions: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
    ### target: This attribute (and all others) are fractional values which represent the fraction of human
    ###     raters who believed the attribute applied to the given comment.
    ###     For evaluation, test set examples with target >= 0.5 will be considered to be in the positive class (toxic).
    delete_these = [
        "id",
        # "target",
        # "comment_text",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        # "asian",
        "atheist",
        "bisexual",
        # "black",
        "buddhist",
        "christian",
        "female",
        "heterosexual",
        "hindu",
        "homosexual_gay_or_lesbian",
        "intellectual_or_learning_disability",
        "jewish",
        "latino",
        "male",
        "muslim",
        "other_disability",
        "other_gender",
        "other_race_or_ethnicity",
        "other_religion",
        "other_sexual_orientation",
        "physical_disability",
        "psychiatric_or_mental_illness",
        "transgender",
        "white",
        "created_date",
        "publication_id",
        "parent_id",
        "article_id",
        "rating",
        "funny",
        "wow",
        "sad",
        "likes",
        "disagree",
        "sexual_explicit",
        "identity_annotator_count",
        "toxicity_annotator_count",
    ]

    df_raw.drop(delete_these, axis=1, inplace=True)
    df_raw.dropna(inplace=True)

    ### delete NAN in columns: "black" and "asian"
    df_raw = df_raw[pd.notnull(df_raw["asian"])]
    df_raw = df_raw[pd.notnull(df_raw["black"])]
    ### drop rows with zero in column "black" and column "asian"
    df_raw = df_raw[(df_raw["black"] != 0) | (df_raw["asian"] != 0)]
    idx_common = (df_raw["black"] != 0) & (df_raw["asian"] != 0)
    df_raw = df_raw[~idx_common]
    ### some rows have both non-zero "black" and non-zero "asian"
    print("#black: ", len(df_raw[df_raw["black"] != 0]))
    print("#asian: ", len(df_raw[df_raw["asian"] != 0]))
    print(
        "#black + #asian: ",
        len(df_raw[df_raw["black"] != 0]) + len(df_raw[df_raw["asian"] != 0]),
    )

    print(
        "#black&asian",
        len(df_raw[(df_raw["asian"] != 0) & (df_raw["black"] != 0)]),
    )

    def get_group(a):
        maj = a.apply(pd.Series.idxmax, axis=1)
        return maj.rename("asian_or_black")

    df_sens = df_raw[["asian", "black"]]
    df_race = get_group(df_sens)
    df_race = df_race.map({"asian": 0, "black": 1})
    df_raw.drop(["asian", "black"], axis=1, inplace=True)
    df_raw = pd.concat([df_raw, df_race], axis=1, ignore_index=False)

    ### conver target into binary
    if split == "test":
        df_raw.rename(columns={"toxicity": "target"}, inplace=True)
    print(df_raw.describe())
    return df_raw


def preprocess_toxic():
    df_raw_tr = pd.read_csv("data/train.csv")
    df_raw_te = pd.read_csv("data/test_public_expanded.csv")
    df_tr = clean_toxic_raw(df_raw_tr, split="train")
    df_tr.to_csv("data/train_cleaned.csv", index=False)
    print("After cleaning, #train={}".format(len(df_tr)))
    df_te = clean_toxic_raw(df_raw_te, split="test")
    df_te.to_csv("data/test_cleaned.csv", index=False)
    print("After cleaning, #test={}".format(len(df_te)))


def get_bert_embedding(str_list):
    from transformers import AutoTokenizer, BertModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased").to(device)
    embedding = []
    for str in tqdm(str_list):
        inputs = tokenizer(str, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embedding.append(outputs["last_hidden_state"][0, 0, :].detach().cpu().numpy())
    return np.stack(embedding)


def get_bert_embedding_toxic():
    ### extract embedding for train
    df = pd.read_csv("data/train_cleaned.csv")
    X = get_bert_embedding(list(df["comment_text"]))
    A = df["asian_or_black"].to_numpy(dtype=np.int)
    Y = df["target"].to_numpy()
    with h5py.File("data/train_bert.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("asian_or_black", data=A)
        f.create_dataset("Y", data=Y)
    ### extract embedding for test
    df = pd.read_csv("data/test_cleaned.csv")
    X = get_bert_embedding(list(df["comment_text"]))
    A = df["asian_or_black"].to_numpy(dtype=np.int)
    Y = df["target"].to_numpy(dtype=np.int)
    with h5py.File("data/test_bert.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("asian_or_black", data=A)
        f.create_dataset("Y", data=Y)


if __name__ == "__main__":
    preprocess_toxic()
    get_bert_embedding_toxic()
