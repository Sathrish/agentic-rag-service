"""NFCorpus (BEIR) download and loading. The dataset is downloaded on first use, never committed."""
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

from . import settings


HF_BASE = "https://huggingface.co/datasets/BeIR"
HF_FILES = {
    "corpus.parquet": f"{HF_BASE}/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{HF_BASE}/nfcorpus/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels/test.tsv": f"{HF_BASE}/nfcorpus-qrels/resolve/main/test.tsv",
}


def _download_from_hf(target: Path):
    """Fetch the BEIR NFCorpus mirror on Hugging Face and write the standard BEIR layout."""
    import pandas as pd

    (target / "qrels").mkdir(parents=True, exist_ok=True)
    for name, url in HF_FILES.items():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target / name)
    for name in ("corpus", "queries"):
        frame = pd.read_parquet(target / f"{name}.parquet")
        frame.to_json(target / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        (target / f"{name}.parquet").unlink()


def download_nfcorpus(data_dir: Path = settings.DATA_DIR) -> Path:
    """The original BEIR server sends an incomplete TLS chain that Python rejects,
    so try it first and fall back to the identical Hugging Face mirror."""
    data_dir = Path(data_dir)
    target = data_dir / "nfcorpus"
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "nfcorpus.zip"
    try:
        if not zip_path.exists():
            print(f"Downloading NFCorpus from {settings.NFCORPUS_URL}")
            urllib.request.urlretrieve(settings.NFCORPUS_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
    except OSError as exc:
        print(f"Original source failed ({exc.__class__.__name__}); using the Hugging Face mirror")
        zip_path.unlink(missing_ok=True)
        _download_from_hf(target)
    return target


def load_nfcorpus(data_dir: Path = settings.DATA_DIR, split: str = "test"):
    """Return ``(corpus, queries, qrels)``.

    corpus  : {doc_id: {"title": str, "text": str}} in corpus.jsonl order
    queries : {query_id: str}, only queries with at least one relevant document
    qrels   : {query_id: {doc_id: grade}}, grade 2 = highly relevant, 1 = relevant
    """
    path = download_nfcorpus(data_dir)
    corpus = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row.get("title", ""), "text": row.get("text", "")}

    qrels = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            if int(score) > 0 and did in corpus:
                qrels.setdefault(qid, {})[did] = int(score)

    queries = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]
    return corpus, queries, qrels


def load_rag_qa(path: Path = settings.ROOT / "eval" / "rag_qa.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
