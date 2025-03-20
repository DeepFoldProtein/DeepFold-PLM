# plmMSA

## Install Packages

```bash
pip install -r requirements.txt

# Include vdbplmalign, procl as submodules.
pip install -e .
```

## Ankh Contrastive

Available on: https://huggingface.co/DeepFoldProtein/Ankh-Large-Contrastive

```python
from procl.model.ankh import AnkhCL
model = AnkhCL.from_pretrained(
    "DeepFoldProtein/Ankh-Large-Contrastive", freeze_base=True, is_scratch=False
)
tokenizer = AutoTokenizer.from_pretrained("DeepFoldProtein/Ankh-Large-Contrastive")
```

## Example Procedure for building plmMSA

### ESM-1b Embedding Generation

Generate embeddings using the ESM-1b model: 🧬
```bash
python scripts/gen_esm1b_embeddings.py -i example_fastas/example.fasta -o example_embedding_path/esm1b -b 1 -d cuda
```

### Ankh-Contrastive Embedding Generation

Generate embeddings using the Ankh-Contrastive model: 🔍
```bash
python scripts/gen_ankh_contrastive_embeddings.py -i example_fastas/example.fasta -o example_embedding_path/ankh_contrastive -b 1 -d cuda
```

### Train & Save Embedding to Faiss Vector Database

Train and save the embeddings to a Faiss vector database: 💾

For Ankh-Contrastive embeddings:
```bash
python scripts/train_and_save_faiss.py -i example_embedding_path/ankh_contrastive -o example_vdb_path/ankh_contrastive -f example_fastas/example.fasta -n 1
```

For ESM-1b embeddings:
```bash
python scripts/train_and_save_faiss.py -i example_embedding_path/esm1b -o example_vdb_path/esm1b -f example_fastas/example.fasta -n 1
```

### Run Faiss Vector Database API

Start the Faiss Vector Database API: 🚀
```bash
python scripts/faiss_api.py
```

### Test Faiss API

Test the Faiss API to ensure it's working correctly: ✅
```bash
python scripts/test_query.py
```

### Cached Embeddings for vdbplmalign (without pooling)

Generate cached embeddings using the ProtT5 model: 🗃️
```bash
python scripts/gen_prott5_embeddings.py -i example_fastas/example.fasta -o example_embedding_path/prott5 -b 1 -d cuda
```

### Run Full Pipeline

Run the full pipeline with the specified query FASTA file: 🔄
```bash
python run_full_pipeline.py --query_fasta example_fastas/example_query.fasta --nprobe 1 --limit 3
```

## Acknowledgements

vdbplmalign is based on the [PLMAlign](https://github.com/maovshao/PLMAlign) repository. Thanks to the authors for their contributions.
