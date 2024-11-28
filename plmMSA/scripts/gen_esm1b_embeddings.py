import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from torch import clamp, sum
from transformers import EsmModel, EsmTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    def esm1b_get_model(esm1b_model_path):
        if esm1b_model_path is not None:
            print("##########################")
            print("Loading cached model from: {}".format(esm1b_model_path))
            print("##########################")
        model = EsmModel.from_pretrained(esm1b_model_path)

        model = model.to(device)
        model = model.eval()
        tokenizer = EsmTokenizer.from_pretrained(esm1b_model_path)
        return model, tokenizer

    # GPU
    device = args.device

    # Model and Tokenizer
    esm1b_model_path = "facebook/esm1b_t33_650M_UR50S"
    model, vocab = esm1b_get_model(esm1b_model_path)

    # batch size
    batch_size = int(args.batch_size)

    # Read FASTA file
    sequences = {}
    with open(args.input_file, "r") as fasta_file:
        for line in fasta_file:
            if line.startswith(">"):
                seq_id = line[1:].strip()  # Get the sequence ID
            else:
                sequences[seq_id] = (
                    sequences.get(seq_id, "") + line.strip()
                )  # Concatenate sequences

    # Convert to DataFrame
    pandas = pd.DataFrame(list(sequences.items()), columns=["sequence_id", "sequence"])

    # HuggingFace DataSet
    dataset = Dataset.from_pandas(pandas)

    def embed_pt(batch):

        seqs = list()
        for i in batch["sequence"]:
            seq = " ".join(list(i))
            seqs.append(seq)

        token_encoding = vocab.batch_encode_plus(
            seqs, add_special_tokens=True, padding="longest"
        )
        input_ids = torch.tensor(token_encoding["input_ids"]).to(device)
        attention_mask = torch.tensor(token_encoding["attention_mask"]).to(device)

        try:
            with torch.no_grad():
                embedding_repr = model(input_ids, attention_mask=attention_mask)
        except RuntimeError:
            print("RuntimeError during embedding. Try lowering batch size. ")

        sentence_embs = embedding_repr.last_hidden_state.clone()
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(sentence_embs.size()).float()
        )
        sequence_embedding = sum(sentence_embs * input_mask_expanded, 1) / clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        for seq_id, embedding in zip(batch["sequence_id"], sequence_embedding):
            torch.save(embedding.cpu(), Path(args.output_folder) / f"{seq_id}.pt")

        return batch

    dataset = dataset.map(embed_pt, batched=True, batch_size=batch_size)


if __name__ == "__main__":
    main()
