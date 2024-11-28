import logging
import os
import shutil
import tempfile
import zipfile

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from vdbplmalign.config import GlobalConfig
from vdbplmalign.pipeline import Pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = Pipeline()


def cleanup_temp_dir(temp_dir: str):
    if os.path.exists(temp_dir):
        logger.info(f"Cleaning up temporary directory {temp_dir}")
        shutil.rmtree(temp_dir)


def save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def create_zip_file(directory: str, output_zip: str):
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                if file != os.path.basename(output_zip):
                    file_path = os.path.join(root, file)
                    logger.info(f"Adding {file_path} to ZIP archive.")
                    zipf.write(file_path, file)


def run_pipeline_generic(
    background_tasks: BackgroundTasks,
    query_fasta: UploadFile,
    target_fasta: UploadFile,
    cutoff: float,
    use_cache: bool,
    full_pipeline: bool,
    nprobe: int = None,
    limit: int = None,
):
    if not query_fasta.filename.endswith(".fasta") or (
        target_fasta and not target_fasta.filename.endswith(".fasta")
    ):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Please upload .fasta files."
        )

    try:
        temp_dir = tempfile.mkdtemp(dir=os.path.join(os.getcwd(), "tmp"))
        query_fasta_path = os.path.join(temp_dir, query_fasta.filename)
        save_upload_file(query_fasta, query_fasta_path)

        target_fasta_path = None
        if target_fasta:
            target_fasta_path = os.path.join(temp_dir, target_fasta.filename)
            save_upload_file(target_fasta, target_fasta_path)

        embedding_path = os.getenv(
            "EMBEDDING_PATH", "/gpfs/database/milvus/datasets/uniref50_t5/datasets/"
        )
        output_fasta = os.path.join(temp_dir, "plm_vdb_raw.fasta")
        output_path = os.path.join(temp_dir, "plm_vdb_output_")

        if use_cache:
            if full_pipeline:
                pipeline.run_full_cached_pipeline(
                    query_fasta=query_fasta_path,
                    embedding_path=embedding_path,
                    output_fasta=output_fasta,
                    output_path=output_path,
                    cutoff=cutoff,
                    nprobe=nprobe,
                    limit=limit,
                )
            else:
                pipeline.run_alignment_cached_pipeline(
                    query_fasta=query_fasta_path,
                    embedding_path=embedding_path,
                    target_fasta=target_fasta_path,
                    output_path=output_path,
                    cutoff=cutoff,
                )
        else:
            if full_pipeline:
                pipeline.run_full_pipeline(
                    query_fasta=query_fasta_path,
                    output_fasta=output_fasta,
                    output_path=output_path,
                    cutoff=cutoff,
                    nprobe=nprobe,
                    limit=limit,
                )
            else:
                pipeline.run_alignment_pipeline(
                    query_fasta=query_fasta_path,
                    target_fasta=target_fasta_path,
                    output_path=output_path,
                    cutoff=cutoff,
                )

        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        create_zip_file(temp_dir, temp_zip.name)

        zip_filename = temp_zip.name

        if not os.path.exists(zip_filename):
            raise FileNotFoundError(f"ZIP file '{zip_filename}' was not created.")

        logger.info("Pipeline executed successfully.")

        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        return zip_filename
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_cached_pipeline/")
def run_cached_pipeline(
    background_tasks: BackgroundTasks,
    cutoff: float = Form(None),
    query_fasta: UploadFile = File(...),
    nprobe: int = Form(...),
    limit: int = Form(...),
):
    zip_filename = run_pipeline_generic(
        background_tasks,
        query_fasta,
        target_fasta=None,  # No target file in this case
        cutoff=cutoff,
        use_cache=True,
        full_pipeline=True,
        nprobe=nprobe,
        limit=limit,
    )
    return FileResponse(zip_filename, filename="results.zip")


@app.post("/run_cached_alignment/")
def run_cached_alignment(
    background_tasks: BackgroundTasks,
    cutoff: float = Form(None),
    query_fasta: UploadFile = File(...),
    target_fasta: UploadFile = File(...),
):
    zip_filename = run_pipeline_generic(
        background_tasks,
        query_fasta,
        target_fasta,
        cutoff=cutoff,
        use_cache=True,
        full_pipeline=False,
    )
    return FileResponse(zip_filename, filename="results.zip")


@app.post("/run_full_pipeline/")
def run_full_pipeline(
    background_tasks: BackgroundTasks,
    cutoff: float = Form(None),
    query_fasta: UploadFile = File(...),
    nprobe: int = Form(...),
    limit: int = Form(...),
):
    zip_filename = run_pipeline_generic(
        background_tasks,
        query_fasta,
        target_fasta=None,
        cutoff=cutoff,
        use_cache=False,
        full_pipeline=True,
        nprobe=nprobe,
        limit=limit,
    )
    return FileResponse(zip_filename, filename="results.zip")


@app.post("/run_full_alignment/")
def run_full_alignment(
    background_tasks: BackgroundTasks,
    cutoff: float = Form(None),
    query_fasta: UploadFile = File(...),
    target_fasta: UploadFile = File(...),
):
    zip_filename = run_pipeline_generic(
        background_tasks,
        query_fasta,
        target_fasta,
        cutoff=cutoff,
        use_cache=False,
        full_pipeline=False,
    )
    return FileResponse(zip_filename, filename="results.zip")


if __name__ == "__main__":
    import uvicorn

    config = GlobalConfig()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", config.vdbplmalign_API_PORT)),
    )
