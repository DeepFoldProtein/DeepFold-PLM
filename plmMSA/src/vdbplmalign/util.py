import asyncio
import functools
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union

import aiofiles
import numba
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from vdbplmalign.config import GlobalConfig
from vdbplmalign.glob import glob_with_braces as glob

config = GlobalConfig()


def read_fasta(fn_fasta: str) -> Tuple[List[str], Dict[str, str]]:

    prot2seq = {}
    for fn_fasta in glob(fn_fasta):
        with open(fn_fasta) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                prot = record.id
                prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq


def read_fasta_vanilla(fn_fasta: str) -> Tuple[List[str], Dict[str, str]]:

    prot2seq = {}
    for fn_fasta in glob(fn_fasta):
        with open(fn_fasta) as handle:
            lines = handle.readlines()
            prot = None
            seq = ""
            for line in lines:
                if line.startswith(">"):
                    if prot is not None:
                        prot2seq[prot] = seq
                    prot = line.strip()[1:]
                    seq = ""
                else:
                    seq += line.strip()
            if prot is not None:
                prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq


def make_parent_dir(path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)


# Copyright (c) 2022 Laboratory of Structural Bioinformatics
#
# This file is modified from [https://github.com/labstructbioinf/pLM-BLAST].
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.


@numba.njit("f4[:,:](f4[:,:], f4[:,:])", nogil=True, fastmath=True, cache=True)
def dot_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.ndim == 2 and Y.ndim == 2
    assert X.shape[1] == Y.shape[1]

    xlen: int = X.shape[0]
    ylen: int = Y.shape[0]
    embdim: int = X.shape[1]

    emb1_normed: np.ndarray = np.ones((xlen, embdim), dtype=np.float32)
    emb2_normed: np.ndarray = np.ones((ylen, embdim), dtype=np.float32)
    density: np.ndarray = np.empty((xlen, ylen), dtype=np.float32)
    # numba does not support sum() args other then first
    emb1_normed = X / 1
    emb2_normed = Y / 1
    density = emb1_normed @ emb2_normed.T
    return density


@numba.njit("f4[:,:](f4[:,:], f4[:,:])", nogil=True, fastmath=True, cache=True)
def embedding_cos_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.ndim == 2 and Y.ndim == 2
    assert X.shape[1] == Y.shape[1]

    xlen: int = X.shape[0]
    ylen: int = Y.shape[0]
    embdim: int = X.shape[1]
    # normalize
    emb1_norm: np.ndarray = np.empty((xlen, 1), dtype=np.float32)
    emb2_norm: np.ndarray = np.empty((ylen, 1), dtype=np.float32)
    emb1_normed: np.ndarray = np.empty((xlen, embdim), dtype=np.float32)
    emb2_normed: np.ndarray = np.empty((ylen, embdim), dtype=np.float32)
    density: np.ndarray = np.empty((xlen, ylen), dtype=np.float32)
    # numba does not support sum() args other then first
    emb1_norm = np.expand_dims(np.sqrt(np.power(X, 2).sum(1)), 1)
    emb2_norm = np.expand_dims(np.sqrt(np.power(Y, 2).sum(1)), 1)
    emb1_normed = X / emb1_norm
    emb2_normed = Y / emb2_norm
    density = emb1_normed @ emb2_normed.T
    return density


def get_prefilter_list(prefilter_result, query_num):
    prefilter_list = []
    got_query = set()
    with open(prefilter_result) as fp:
        for line in tqdm(fp, desc="Load Result"):
            line_list = line.strip().split("\t")
            protein1 = line_list[0].split(".pdb")[0]
            got_query.add(protein1)
            if len(got_query) > query_num:
                break
            protein2 = line_list[1].split(".pdb")[0]
            score = eval(line_list[2])
            prefilter_list.append(((protein1, protein2), score))
    return prefilter_list


def filter_result_dataframe(
    data: pd.DataFrame, column: Union[str, List[str]] = ["score"]
) -> pd.DataFrame:
    """
    keep spans with biggest score and len
    Args:
            data: (pd.DataFrame)
    Returns:
            filtred frame sorted by score
    """
    data = data.sort_values(by=["len"], ascending=False)
    indices = data.indices.tolist()
    data["y1"] = [yx[0][0] for yx in indices]
    data["x1"] = [yx[0][1] for yx in indices]
    data["score"] = data["score"].round(3)

    if isinstance(column, str):
        column = [column]
    resultsflt = list()
    iterator = data.groupby(["y1", "x1"])
    for col in column:
        for groupid, group in iterator:
            tmp = group.nlargest(1, [col], keep="first")
            resultsflt.append(tmp)
    resultsflt = pd.concat(resultsflt)
    # drop duplicates sometimes
    resultsflt = resultsflt.drop_duplicates(subset=["pathid", "i", "len", "score"])
    # filter
    resultsflt = resultsflt.sort_values(by=["score"], ascending=False)
    return resultsflt


def embedding_load(fasta, embedding_path):
    _, sequences = read_fasta(fasta)
    embedding_result_dic = {}
    for single_sequence in sequences:
        embedding_path = Path(embedding_path)
        embedding_file = embedding_path / f"{single_sequence}.pt"
        embedding_result_dic[single_sequence] = torch.load(embedding_file)
    return embedding_result_dic


async def load_embedding(embedding_file, is_cache: bool):
    async with aiofiles.open(embedding_file, "rb") as f:
        data = await f.read()
        buffer = io.BytesIO(data)
        if is_cache:
            embedding = torch.load(buffer)[:-1, :]
        else:
            embedding = torch.load(buffer)
    return embedding


async def load_single_embedding(
    single_sequence: str,
    embedding_path: Path,
    sequences: Dict[str, str],
    is_cache: bool,
) -> Tuple[str, torch.Tensor, str]:
    """Load a single embedding from the filesystem asynchronously."""
    single_sequence_name = single_sequence + ".pt"
    try:
        embedding_file = embedding_path / single_sequence_name

        embedding = await load_embedding(
            embedding_file=embedding_file, is_cache=is_cache
        )

        return single_sequence, embedding, sequences[single_sequence]
    except Exception as e:
        print(f"An error occurred while loading embedding for {single_sequence}: {e}")
        return single_sequence, None, None


async def load_embedding_from_cache(
    fasta: str, embedding_path: str, is_cache: bool = False
) -> Tuple[Dict[str, str], Dict[str, torch.Tensor]]:
    """Load embeddings for all sequences in a FASTA file using asyncio."""
    keys, sequences = read_fasta_vanilla(fasta)
    embedding_path = Path(embedding_path)

    embedding_result_dict = {}
    valid_sequences = {}

    tasks = [
        load_single_embedding(seq, embedding_path, sequences, is_cache=is_cache)
        for seq in keys
    ]

    for task in asyncio.as_completed(tasks):
        single_sequence, embedding, valid_sequence = await task
        if embedding is not None and valid_sequence is not None:
            embedding_result_dict[single_sequence] = embedding
            valid_sequences[single_sequence] = valid_sequence

    return valid_sequences, embedding_result_dict


def draw_alignment(
    coords: List[Tuple[int, int]], seq1: str, seq2: str, output: Union[None, str]
) -> str:
    """
    draws alignment based on input coordinates
    Args:
            coords: (list) result of align list of tuple indices
            seq1: (str) full residue sequence
            seq2: (str) full residue sequence
            output: (str or bool) if None output is printed
    """
    assert isinstance(seq1, str) or isinstance(
        seq1[0], str
    ), "seq1 must be sting like type"
    assert isinstance(seq2, str) or isinstance(
        seq1[0], str
    ), "seq2 must be string like type"
    assert len(seq1) > 1 and len(seq2), "seq1 or seq1 is too short"

    # check whether alignment indices exeed sequence len
    last_position = coords[-1]
    lp1, lp2 = last_position[0], last_position[1]
    if lp1 >= len(seq1):
        raise KeyError(
            f"mismatch between seq1 length and coords {lp1} - {len(seq1)} for seq2 {lp2} - {len(seq2)}"
        )
    if lp2 >= len(seq2):
        print(seq2)
        raise KeyError(f"mismatch between seq1 length and coords {lp2} - {len(seq2)}")

    # container
    alignment = dict(up=[], relation=[], down=[])
    c1_prev, c2_prev = -1, -1

    for c1, c2 in coords:
        # check if gap occur
        up_increment = True if c1 != c1_prev else False
        down_increment = True if c2 != c2_prev else False

        if up_increment:
            up = seq1[c1]
        else:
            up = "-"

        if down_increment:
            down = seq2[c2]
        else:
            down = "-"

        if up_increment and down_increment:
            relation = "|"
        else:
            relation = " "

        alignment["up"].append(up)
        alignment["relation"].append(relation)
        alignment["down"].append(down)

        c1_prev = c1
        c2_prev = c2
    # merge into 3 line string
    if output != "html":
        string = "".join(alignment["up"]) + "\n"
        string += "".join(alignment["relation"]) + "\n"
        string += "".join(alignment["down"])
        if output is not None:
            return string
        else:
            print(string)


def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv("LOG_TIME") == "1":
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
            return result
        else:
            return func(*args, **kwargs)

    return wrapper
