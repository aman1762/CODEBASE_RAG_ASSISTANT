import os
import shutil
import zipfile
import nbformat
from git import Repo
from chunker import chunk_code

SUPPORTED_EXTENSIONS = (".py", ".js", ".java", ".cpp", ".txt")
BASE_REPO_DIR = "/tmp/user_repo"
BASE_ZIP_DIR = "/tmp/user_zip"

def load_repo(repo_url: str) -> str:
    if os.path.exists(BASE_REPO_DIR):
        shutil.rmtree(BASE_REPO_DIR)
    Repo.clone_from(repo_url, BASE_REPO_DIR)
    return BASE_REPO_DIR

def extract_zip(zip_file) -> str:
    if os.path.exists(BASE_ZIP_DIR):
        shutil.rmtree(BASE_ZIP_DIR)
    os.makedirs(BASE_ZIP_DIR, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(BASE_ZIP_DIR)

    return BASE_ZIP_DIR

def ingest_repo(repo_path: str):
    documents = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".ipynb"):
                documents.extend(parse_notebook(path))

            elif file.endswith(SUPPORTED_EXTENSIONS):
                try:
                    with open(path, "r", errors="ignore") as f:
                        code = f.read()
                        documents.extend(chunk_code(path, code))
                except Exception:
                    pass

    return documents

def parse_notebook(file_path: str):
    docs = []
    try:
        nb = nbformat.read(file_path, as_version=4)
        code_cells = [
            cell.source for cell in nb.cells if cell.cell_type == "code"
        ]
        combined = "\n\n".join(code_cells)
        if len(combined.strip()) > 100:
            docs.extend(chunk_code(file_path, combined))
    except Exception:
        pass
    return docs
