import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from procl.model.ankh import AnkhCL
from torch import clamp, sum
from transformers import T5EncoderModel, T5Tokenizer


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

    def prottrans_get_t5_model(prottrans_model_path):
        if prottrans_model_path is not None:
            print("##########################")
            print("Loading cached model from: {}".format(prottrans_model_path))
            print("##########################")
        model = T5EncoderModel.from_pretrained(prottrans_model_path)

        model = model.to(device)
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained(
            prottrans_model_path, do_lower_case=False
        )
        return model, tokenizer

    # GPU
    device = args.device

    # Model and Tokenizer
    prottrans_model_path = "Rostlab/prot_t5_xl_uniref50"
    model, vocab = prottrans_get_t5_model(prottrans_model_path)

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

    def apply_mask(tensor, mask):
        lengths = mask.sum(dim=1)

        # 각 데이터의 원래 길이를 기준으로 텐서를 잘라서 새로운 리스트에 저장
        masked_tensors = []
        for i, length in enumerate(lengths):
            masked_tensors.append(tensor[i, :length, :])  # 가장 마지막 토큰은 제외

        # 잘린 텐서들을 쌓아서 새로운 리스트 생성
        result = masked_tensors

        return result

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

        before_pooling = apply_mask(sentence_embs, attention_mask)

        for seq_id, embedding in zip(batch["sequence_id"], before_pooling):
            torch.save(embedding.cpu(), Path(args.output_folder) / f"{seq_id}.pt")

        return batch

    dataset = dataset.map(embed_pt, batched=True, batch_size=batch_size)


if __name__ == "__main__":
    main()
