# plmMSA

## Example Procedure


### Build Vector Database

#### ESM-1b embedding generation

```bash
python scripts/gen_esm1b_embeddings.py -i example_fastas/example.fasta -o example_embedding_path/esm1b -b 1 -d cuda
```
#### Ankh-Contrastive embedding generation

```bash
python scripts/gen_ankh_contrastive_embeddings.py -i example_fastas/example.fasta -o example_embedding_path/ankh_contrastive -b 1 -d cuda
```


#### Train & Save Embedding to Faiss Vector Database

```bash
python scripts/train_and_save_faiss.py -i example_embedding_path/ankh_contrastive -o example_vdb_path/ankh_contrastive -f example_fastas/example.fasta -n 1
```
```bash
python scripts/train_and_save_faiss.py -i example_embedding_path/esm1b -o example_vdb_path/esm1b -f example_fastas/example.fasta -n 1
```



#### Run Faiss Vector Database API


```bash
python scripts/faiss_api.py
```

test faiss api

```bash
python scripts/test_query.py
```



### Cached Embeddings for vdbplmalign (without pooling)


```bash
python scripts/gen_prott5_embeddings.py -i example_fastas/example.fasta -o example_embedding_path/prott5 -b 1 -d cuda
```


### Run Full Pipeline

```bash
python run_full_pipeline.py --query_fasta  example_fastas/example_query.fasta --nprobe  1 --limit 3
```
