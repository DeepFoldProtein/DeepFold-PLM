# plmMSA

## API Access 🚀 (Experimental)

The easiest way to use plmMSA is through our web API. This allows you to generate MSAs without setting up the entire pipeline locally. Our API is compatible with mmseqs2 format and methodologies, providing seamless integration with existing mmseqs2 workflows and supporting standard protein sequence formats.

### Submit MSA Job

Submit a job to generate MSAs for your protein sequences:

```bash
curl -X POST 'https://df-plm.deepfold.org/api/plmmsa/v1/submit' \
-H 'Content-Type: application/json' \
-d '{
    "mode": "unpaired+paired", 
    "sequences": [
        "MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA",
        "MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA"
    ]
}'
```

### Check Job Status

Check the status of your submitted job using the job ID returned from the submission:

```bash
curl -X GET 'https://df-plm.deepfold.org/api/plmmsa/v1/job/YOUR_JOB_ID'
```

### MSA Generation Modes

The API supports three different modes for MSA generation:

- **`unpaired`**: Generates MSAs for each sequence independently. Creates separate MSAs for each input sequence and combines them.

- **`paired`**: Designed for multiple related sequences (e.g., different chains of a protein complex). Attempts to find paired homologs for all input sequences together to maintain relationships between sequences.

- **`unpaired+paired`**: Combines both approaches. First attempts to generate a paired MSA, then supplements with unpaired MSAs if the maximum number of sequences isn't reached. Provides the most comprehensive MSA by leveraging both paired and unpaired approaches.

## Integration with Structure Prediction Tools 🧬

plmMSA outputs are compatible with popular structure prediction tools for enhanced folding accuracy.

**Easy Integration with ColabFold:**
```python
results = run(
    queries=queries,
    result_dir=result_dir,
    use_templates=use_templates,
    ...  # other parameters
    host_url="https://df-plm.deepfold.org/api/plmmsa"
)
```

**Easy Integration with Boltz:**
```bash
boltz predict protein.yaml --use_msa_server --msa_server_url "https://df-plm.deepfold.org/api/plmmsa"
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
