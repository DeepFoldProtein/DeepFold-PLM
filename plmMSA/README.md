# plmMSA

## API Access 🚀 (Experimental)

The easiest way to use plmMSA is through our hosted web API. This allows you to generate MSAs without setting up the entire pipeline locally. The API also exposes ColabFold/MMseqs2-compatible endpoints, providing seamless integration with existing MMseqs2 workflows and supporting standard protein sequence formats.

The full, interactive API reference (Swagger UI) is available at **[https://plmmsa.deepfold.org/docs](https://plmmsa.deepfold.org/docs)**.

### Submit MSA Job

Submit a job to generate MSAs for your protein sequences. The endpoint returns `202 Accepted` with a `job_id`:

```bash
curl -X POST 'https://plmmsa.deepfold.org/v2/msa' \
-H 'Content-Type: application/json' \
-d '{
    "sequences": [
        "MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA",
        "MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA"
    ],
    "paired": true,
    "output_format": "a3m"
}'
```

### Check Job Status

Check the status of your submitted job (and retrieve results once complete) using the `job_id` returned from the submission:

```bash
curl -X GET 'https://plmmsa.deepfold.org/v2/msa/YOUR_JOB_ID'
```

### Key Request Parameters

`POST /v2/msa` accepts the following main fields (see the Swagger UI for the full schema):

- **`sequences`** (required): One string per chain. Provide multiple sequences for a protein complex.
- **`paired`** (default `false`): When `true`, produces a paired MSA across chains (in addition to per-chain unpaired MSAs), preserving the pairing relationships between chains of a complex.
- **`output_format`** (default `"a3m"`): Wire format for the returned MSA.
- **`models`**: One or more PLM backend IDs to run (available: `ankh_cl`, `ankh_large`, `esm1b`, `prott5`).
- **`mode`**: Alignment mode — `local`, `global`, or `glocal`/`q2t`/`t2q` (OTalign-only).

> **Note:** The legacy `v1` API (`/api/plmmsa/v1/...`) has been retired. Use the `v2` endpoints shown above.

## Integration with Structure Prediction Tools 🧬

plmMSA outputs are compatible with popular structure prediction tools for enhanced folding accuracy.

**Easy Integration with ColabFold:**
```python
results = run(
    queries=queries,
    result_dir=result_dir,
    use_templates=use_templates,
    ...  # other parameters
    host_url="https://plmmsa.deepfold.org/v2/colabfold/plmmsa"
)
```

**Easy Integration with Boltz:**
```bash
boltz predict protein.yaml --use_msa_server --msa_server_url "https://plmmsa.deepfold.org/v2/colabfold/plmmsa"
```


## Manual Setup: Example Procedure for building plmMSA

### Install Packages

```bash
pip install -r requirements.txt

# Include vdbplmalign, procl as submodules.
pip install -e .
```

### Ankh Contrastive

Available on: https://huggingface.co/DeepFoldProtein/Ankh-Large-Contrastive

```python
from procl.model.ankh import AnkhCL
model = AnkhCL.from_pretrained(
    "DeepFoldProtein/Ankh-Large-Contrastive", freeze_base=True, is_scratch=False
)
tokenizer = AutoTokenizer.from_pretrained("DeepFoldProtein/Ankh-Large-Contrastive")
```

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
