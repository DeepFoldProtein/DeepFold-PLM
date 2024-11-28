import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from vdbplmalign.util import log_time

from .embedding_generate import prottrans_embedding_generate
from .plmalign_util.alignment import (
    vdbplmalign_gather_all_paths,
    vdbplmalign_search_paths,
)
from .plmblast_util.alignment import plmblast_gather_all_paths, plmblast_search_paths
from .util import (
    dot_product,
    draw_alignment,
    embedding_cos_similarity,
    embedding_load,
    filter_result_dataframe,
    load_embedding_from_cache,
    make_parent_dir,
    read_fasta,
)


class vdbplmalign:
    """
    main class for handling alignment extaction
    """

    NORM: Union[bool, str] = True
    MODE: str = "local"
    GAP_EXT: float = 1.0

    def __init__(self, *args, **kw_args):
        pass

    def embedding_to_span(self, X, Y, result_mode: str = "results") -> pd.DataFrame:

        ### vdbplmalign (dot)
        X = X.numpy()
        Y = Y.numpy()

        # dot_product 함수 실행 시간 측정 시작
        densitymap = dot_product(X, Y)

        densitymap = densitymap.T

        path = vdbplmalign_gather_all_paths(
            densitymap,
            norm=self.NORM,
            mode=self.MODE,
            gap_extension=self.GAP_EXT,
            with_scores=True if result_mode == "all" else False,
        )

        if result_mode == "all":
            scorematrix = path[1]
            path = path[0]

        results = vdbplmalign_search_paths(
            densitymap, path=path, mode=self.MODE, as_df=True
        )

        if result_mode == "all":
            return (results, densitymap, path, scorematrix)
        else:
            return results


class plmblast:
    """
    main class for handling alignment extaction
    """

    MIN_SPAN_LEN: int = 20
    WINDOW_SIZE: int = 20
    NORM: Union[bool, str] = True
    BFACTOR: str = "local"
    SIGMA_FACTOR: float = 1
    GAP_OPEN: float = 0.0
    GAP_EXT: float = 0.0
    FILTER_RESULTS: bool = False

    def __init__(self, *args, **kw_args):
        pass

    def embedding_to_span(self, X, Y, mode: str = "results") -> pd.DataFrame:

        ### pLM-BLAST (cos)
        X = X.numpy()
        Y = Y.numpy()
        densitymap = embedding_cos_similarity(X, Y)

        densitymap = densitymap.T

        paths = plmblast_gather_all_paths(
            densitymap,
            norm=self.NORM,
            minlen=self.MIN_SPAN_LEN,
            bfactor=self.BFACTOR,
            gap_opening=self.GAP_OPEN,
            gap_extension=self.GAP_EXT,
            with_scores=True if mode == "all" else False,
        )
        if mode == "all":
            scorematrix = paths[1]
            paths = paths[0]
        results = plmblast_search_paths(
            densitymap,
            paths=paths,
            window=self.WINDOW_SIZE,
            min_span=self.MIN_SPAN_LEN,
            sigma_factor=self.SIGMA_FACTOR,
            mode=self.BFACTOR,
            as_df=True,
        )
        if mode == "all":
            return (results, densitymap, paths, scorematrix)
        else:
            return results

    def full_compare(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        idx: int = 0,
        file: str = "source.fasta",
    ) -> pd.DataFrame:
        """
        Args:
            emb1: (np.ndarray) sequence embedding [seqlen x embdim]
            emb2: (np.ndarray) sequence embedding [seqlen x embdim]
            idx: (int) identifier used when multiple function results are concatenated
            file: (str) embedding/sequence source file may be omitted
        Returns:
            data: (pd.DataFrame) frame with alignments and their scores
        """
        res = self.embedding_to_span(emb1, emb2)
        if len(res) > 0:
            # add referece index to each hit
            res["i"] = idx
            res["dbfile"] = file
            # filter out redundant hits
            if self.FILTER_RESULTS:
                res = filter_result_dataframe(res, column="score")
        return res


def pairwise_align(embedding1, embedding2, seq1, seq2, mode, method="vdbplmalign"):
    if method == "vdbplmalign":
        extr = vdbplmalign()
        extr.MODE = mode
        results = extr.embedding_to_span(embedding2, embedding1)
    elif method == "plmblast":
        extr = plmblast()
        extr.BFACTOR = mode
        extr.FILTER_RESULTS = True
        if mode == "local":
            results = extr.full_compare(embedding2, embedding1)
            if len(results) == 0:
                extr.BFACTOR = "global"
                results = extr.embedding_to_span(embedding2, embedding1)
                extr.BFACTOR = "local"
        else:
            results = extr.embedding_to_span(embedding2, embedding1)
    else:
        assert method in {"vdbplmalign", "plmblast"}

    row = results.iloc[0]
    aln = draw_alignment(row.indices, seq1, seq2, output="str")
    return row["score"].item(), aln


def align_task(args):
    (
        single_query,
        single_target,
        query_embeddings,
        target_embeddings,
        query_sequences,
        target_sequences,
        mode,
        method,
    ) = args

    try:
        score, results = pairwise_align(
            query_embeddings[single_query],
            target_embeddings[single_target],
            query_sequences[single_query],
            target_sequences[single_target],
            mode,
            method=method,
        )

        return single_query, single_target, score, results

    except Exception as e:
        print(e)
        print(f"An error occurred while align {single_query}")
        return None, None, None, None


def prottrans_get_t5_model(prottrans_model_path, prottrans_device):
    if prottrans_model_path is not None:
        print("##########################")
        print("Loading cached model from: {}".format(prottrans_model_path))
        print("##########################")
    model = T5EncoderModel.from_pretrained(prottrans_model_path)
    print(prottrans_device)
    model = model.to(prottrans_device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(prottrans_model_path, do_lower_case=False)
    return model, tokenizer


def warmup_porttrans_model(model, tokenizer, prottrans_device):
    inputs = tokenizer("AATTTTTT", return_tensors="pt")
    model(
        inputs["input_ids"].to(prottrans_device),
        inputs["attention_mask"].to(prottrans_device),
    )


@log_time
def execute_alignment_tasks(
    tasks, query_sequences, target_sequences, if_stdout, output_path, align_format
):
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(align_task, task): task for task in tasks}
        protein_pair_dict = {protein: [] for protein in query_sequences}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Alignments"):
            single_query, single_target, score, results = future.result()
            if score:
                if if_stdout:
                    print(f"{single_query}\t{single_target}\t Score = {score}\n")
                    print(f"{single_query}\t{single_target}\n{results}\n")
                if output_path is not None:
                    protein_pair_dict[single_query].append(
                        (single_target, score, results)
                    )

    if output_path is not None:
        output_score = output_path + "score"
        make_parent_dir(output_score)
        output_alignment = output_path + "alignment"
        f1 = open(output_score, "w")
        f2 = open(output_alignment, "w")
        output_score_sort = output_path + "score_sort"

        for query_protein in query_sequences:
            protein_pair_dict[query_protein] = sorted(
                protein_pair_dict[query_protein], key=lambda x: x[1], reverse=True
            )

        for query_protein in query_sequences:
            for target_protein, score, results in protein_pair_dict[query_protein]:
                f1.write(f"{query_protein}\t{target_protein}\t{score}\n")
                if align_format == "fasta":
                    f2.write(
                        f">{query_protein}\t{target_protein}\t{score:8.3f}\n{results}\n"
                    )
                else:
                    f2.write(
                        f"{query_protein}\t{target_protein}\t{score:8.3f}\n{results}\n"
                    )

        f1.close()
        f2.close()

        with open(output_score_sort, "w") as f3:
            for query_protein in query_sequences:
                for target_protein, score, _ in protein_pair_dict[query_protein]:
                    f3.write(f"{query_protein}\t{target_protein}\t{score}\n")


@log_time
def vdbplmalign_cache_pipeline(
    query_fasta,
    target_fasta,
    embedding_path,
    prottrans_model,
    prottrans_tokenizer,
    prottrans_device,
    query_embedding_path=None,
    target_embedding_path=None,
    mode="global",
    search_result_setting=None,
    output_path=None,
    if_stdout=True,
    align_format="fasta",
):

    method = "vdbplmalign"
    print(f"Align with method: {method}")

    query_id, query_sequences = read_fasta(query_fasta)
    query_id = query_id[0]
    _, target_sequences = read_fasta(target_fasta)
    query_prottrans_start_time = time.time()
    query_embeddings = prottrans_embedding_generate(
        query_fasta, prottrans_model, prottrans_tokenizer, device=prottrans_device
    )
    query_prottrans_end_time = time.time()
    logging.info(
        f"Query prottrans time: {query_prottrans_end_time - query_prottrans_start_time} seconds"
    )
    target_load_start_time = time.time()
    target_sequences, target_embeddings = asyncio.run(
        load_embedding_from_cache(target_fasta, embedding_path, is_cache=True)
    )
    target_sequences[query_id] = query_sequences[query_id]
    target_embeddings[query_id] = query_embeddings[query_id]
    target_load_end_time = time.time()
    logging.info(
        f"Embedding load time: {target_load_end_time - target_load_start_time} seconds"
    )

    tasks = []

    if search_result_setting is None:
        for single_query in query_sequences:
            for single_target in target_sequences:
                tasks.append(
                    (
                        single_query,
                        single_target,
                        query_embeddings,
                        target_embeddings,
                        query_sequences,
                        target_sequences,
                        mode,
                        method,
                    )
                )
    else:
        search_result_path = search_result_setting[0]
        top = search_result_setting[1]

        with open(search_result_path, "r") as f:
            pairs = f.readlines()

        for line in pairs:
            single_query, single_target, similarity = line.strip().split()
            similarity = eval(similarity)

            if (top is not None) and (similarity < top):
                continue

            tasks.append(
                (
                    single_query,
                    single_target,
                    query_embeddings,
                    target_embeddings,
                    query_sequences,
                    target_sequences,
                    mode,
                    method,
                )
            )
    align_start_time = time.time()
    execute_alignment_tasks(
        tasks, query_sequences, target_sequences, if_stdout, output_path, align_format
    )
    align_end_time = time.time()
    logging.info(f"Alignment time: {align_end_time - align_start_time} seconds")


@log_time
def vdbplmalign_pipeline(
    query_fasta,
    target_fasta,
    prottrans_model,
    prottrans_tokenizer,
    prottrans_device,
    mode="global",
    embedding_path=None,
    query_embedding_path=None,
    target_embedding_path=None,
    search_result_setting=None,
    output_path=None,
    if_stdout=False,
):
    method = "vdbplmalign"
    print(f"Align with method: {method}")

    _, query_sequences = read_fasta(query_fasta)
    _, target_sequences = read_fasta(target_fasta)

    if query_embedding_path == None:
        query_embeddings = prottrans_embedding_generate(
            query_fasta,
            prottrans_model=prottrans_model,
            prottrans_tokenizer=prottrans_tokenizer,
            device=prottrans_device,
        )
    else:
        query_embeddings = embedding_load(query_fasta, query_embedding_path)

    if target_embedding_path == None:
        target_embeddings = prottrans_embedding_generate(
            target_fasta,
            prottrans_model=prottrans_model,
            prottrans_tokenizer=prottrans_tokenizer,
            device=prottrans_device,
        )
    else:
        target_embeddings = embedding_load(target_fasta, target_embedding_path)

    if output_path != None:
        output_score = output_path + "score"
        make_parent_dir(output_score)
        output_alignment = output_path + "alignment"
        f1 = open(output_score, "w")
        f2 = open(output_alignment, "w")

        protein_pair_dict = {}
        for protein in query_sequences:
            protein_pair_dict[protein] = []
        output_score_sort = output_path + "score_sort"

    if search_result_setting == None:
        for single_query in tqdm(query_sequences, desc="Query"):
            for single_target in target_sequences:
                score, results = pairwise_align(
                    query_embeddings[single_query],
                    target_embeddings[single_target],
                    query_sequences[single_query],
                    target_sequences[single_target],
                    mode,
                    method=method,
                )
                if if_stdout:
                    print(f"{single_query}\t{single_target}\t Score = {score}\n")
                    print(f"{single_query}\t{single_target}\n{results}\n")
                if output_path != None:
                    f1.write(f"{single_query}\t{single_target}\t{score:8.3f}\n")
                    f2.write(
                        f"{single_query}\t{single_target}\t{score:8.3f}\n{results}\n"
                    )
                    protein_pair_dict[single_query].append((single_target, score))

    else:
        search_result_path = search_result_setting[0]
        top = search_result_setting[1]

        with open(search_result_path, "r") as f:
            pairs = f.readlines()

        for line in tqdm(pairs, desc="Search result"):
            single_query, single_target, similarity = line.strip().split()
            similarity = eval(similarity)

            if (top != None) and (similarity < top):
                continue

            score, results = pairwise_align(
                query_embeddings[single_query],
                target_embeddings[single_target],
                query_sequences[single_query],
                target_sequences[single_target],
                mode,
                method=method,
            )
            if if_stdout:
                print(f"{single_query}\t{single_target}\t Score = {score}\n")
                print(f"{single_query}\t{single_target}\n{results}\n")
            if output_path != None:
                f1.write(f"{single_query}\t{single_target}\t{score:8.3f}\n")
                f2.write(f"{single_query}\t{single_target}\t{score:8.3f}\n{results}\n")
                protein_pair_dict[single_query].append((single_target, score))

    if output_path != None:
        f1.close()
        f2.close()

        for query_protein in query_sequences:
            protein_pair_dict[query_protein] = sorted(
                protein_pair_dict[query_protein], key=lambda x: x[1], reverse=True
            )

        with open(output_score_sort, "w") as f3:
            for query_protein in query_sequences:
                for pair in protein_pair_dict[query_protein]:
                    f3.write(f"{query_protein}\t{pair[0]}\t{pair[1]}\n")
