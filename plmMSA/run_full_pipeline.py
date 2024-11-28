import argparse
import os

from vdbplmalign.config import GlobalConfig
from vdbplmalign.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline with specified parameters."
    )
    parser.add_argument(
        "--query_fasta", type=str, required=True, help="Path to the query FASTA file."
    )
    parser.add_argument(
        "--nprobe", type=int, required=True, help="Number of probes for FAISS."
    )
    parser.add_argument(
        "--limit", type=int, required=True, help="Limit for the number of results."
    )

    args = parser.parse_args()
    query_fasta_dir = os.path.dirname(args.query_fasta)

    GlobalConfig().reinit(
        ankh_device="cuda:1",
        esm1b_device="cuda:2",
        prottrans_device="cuda:3",
        vdbplmalign_api_port=None,
        faiss_api_url="http://localhost:31818/search",
        destroy_models_after_search=False,
    )
    output_fasta = os.path.join(
        query_fasta_dir, "msas", "vdb_ankh_esm1b_faiss", "plm_vdb_raw.fasta"
    )
    output_path_prefix = os.path.join(
        query_fasta_dir, "msas", "vdb_ankh_esm1b_faiss", "plm_vdb_output_"
    )

    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    pipeline = Pipeline()

    pipeline._run_pipeline(
        query_fasta=args.query_fasta,
        output_fasta=output_fasta,
        target_fasta=None,
        output_path=output_path_prefix,
        full_pipeline=True,
        nprobe=args.nprobe,
        limit=args.limit,
        use_cache=True,
        embedding_path="example_embedding_path/prott5",
    )


if __name__ == "__main__":
    main()
