import json
import logging
import os
import time

from Bio import SeqIO
from vdbplmalign.alignment_to_a3m import alignment_to_a3m
from vdbplmalign.config import GlobalConfig
from vdbplmalign.plmalign import (
    prottrans_get_t5_model,
    vdbplmalign_cache_pipeline,
    vdbplmalign_pipeline,
    warmup_porttrans_model,
)
from vdbplmalign.util import log_time
from vdbplmalign.vdb.fetch_vdb import FetchVdb
from vdbplmalign.vdb.utils import combine_and_generate_fasta

config = GlobalConfig()


def extract_sequence_from_fasta(query_fasta):
    with open(query_fasta, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.id), str(record.seq)


class Pipeline:
    def __init__(self, prottrans_model_path="Rostlab/prot_t5_xl_uniref50"):
        self.prottrans_model_path = prottrans_model_path
        self.prottrans_model, self.prottrans_tokenizer = None, None
        self.prottrans_device = config.PROTTRANS_DEVICE
        self.fetch_vdb = self._setup_vdb_fetch()
        self.load_prottrans_model()

    def _setup_vdb_fetch(self):
        return FetchVdb(
            ankh_device=config.ANKH_DEVICE,
            esm1b_device=config.ESM1B_DEVICE,
        )

    def load_prottrans_model(self):
        if self.prottrans_model_path:
            self.prottrans_model, self.prottrans_tokenizer = prottrans_get_t5_model(
                self.prottrans_model_path, self.prottrans_device
            )
            warmup_porttrans_model(
                self.prottrans_model, self.prottrans_tokenizer, self.prottrans_device
            )

    def search_uniref(self, query_fasta, nprobe, limit):
        _, sequence = extract_sequence_from_fasta(query_fasta=query_fasta)
        return self.fetch_vdb.search_models(sequence, nprobe=nprobe, limit=limit)

    def _run_pipeline(
        self,
        query_fasta,
        output_fasta=None,
        target_fasta=None,
        mode="global",
        query_embedding_path=None,
        target_embedding_path=None,
        embedding_path=None,
        search_result_setting=None,
        output_path=None,
        if_stdout=False,
        cutoff=None,
        use_cache=False,
        full_pipeline=True,
        nprobe=None,
        limit=None,
    ):
        global_start_time = time.time()

        if not self.prottrans_model or not self.prottrans_tokenizer:
            raise ValueError(
                "ProtTrans 모델이 로드되지 않았습니다. `load_prottrans_model` 메서드를 사용하여 로드하세요."
            )

        if full_pipeline:
            search_start_time = time.time()
            model_results: dict = self.search_uniref(
                query_fasta, nprobe=nprobe, limit=limit
            )
            output_json = output_path + "vdb_results.json"
            with open(output_json, "w") as f:
                json.dump(model_results, f)
            combine_and_generate_fasta(
                input_fasta=query_fasta,
                output_fasta=output_fasta,
                model_results=model_results,
            )
            search_end_time = time.time()
            logging.info(f"Search time: {search_end_time - search_start_time} seconds")

        target_fasta_to_use = output_fasta if full_pipeline else target_fasta
        pipeline = vdbplmalign_cache_pipeline if use_cache else vdbplmalign_pipeline

        pipeline_start_time = time.time()
        pipeline(
            query_fasta=query_fasta,
            target_fasta=target_fasta_to_use,
            mode=mode,
            query_embedding_path=query_embedding_path,
            target_embedding_path=target_embedding_path,
            embedding_path=embedding_path,
            prottrans_model=self.prottrans_model,
            prottrans_tokenizer=self.prottrans_tokenizer,
            prottrans_device=self.prottrans_device,
            search_result_setting=search_result_setting,
            output_path=output_path,
            if_stdout=if_stdout,
        )
        pipeline_end_time = time.time()
        logging.info(
            f"Pipeline time: {pipeline_end_time - pipeline_start_time} seconds"
        )

        alignment_path = output_path + "alignment"
        a3m_path = output_path + "alignment.a3m"
        alignment_to_a3m(alignment_path, a3m_path, cutoff=cutoff)
        alignment_to_a3m_end_time = time.time()
        logging.info(
            f"Alignment to a3m time: {alignment_to_a3m_end_time - pipeline_end_time} seconds"
        )
        global_end_time = time.time()
        logging.info(f"Global time: {global_end_time - global_start_time} seconds")

    @log_time
    def run_full_cached_pipeline(self, *args, **kwargs):
        self._run_pipeline(*args, **kwargs, use_cache=True, full_pipeline=True)

    @log_time
    def run_alignment_cached_pipeline(self, *args, **kwargs):
        self._run_pipeline(*args, **kwargs, use_cache=True, full_pipeline=False)

    @log_time
    def run_full_pipeline(self, *args, **kwargs):
        self._run_pipeline(*args, **kwargs, use_cache=False, full_pipeline=True)

    @log_time
    def run_alignment_pipeline(self, *args, **kwargs):
        self._run_pipeline(*args, **kwargs, use_cache=False, full_pipeline=False)
