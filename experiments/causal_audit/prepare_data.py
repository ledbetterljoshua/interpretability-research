"""Download and deterministically select ARC-Easy development data; no model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/causal_audit"
SEED = 731


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "development.json"
    if dest.exists():
        raise FileExistsError("Preserve the existing pinned dataset")
    repo = "allenai/ai2_arc"
    revision = HfApi().dataset_info(repo).sha
    result = dict(dataset=repo, revision=revision, config="ARC-Easy", seed=SEED,
                  license="CC-BY-SA-4.0", source_files={}, splits={})
    for split, n in [("train", 128), ("validation", 64)]:
        filename = f"ARC-Easy/{split}-00000-of-00001.parquet"
        path = Path(hf_hub_download(repo, filename, repo_type="dataset",
                                   revision=revision, cache_dir=OUT / "cache"))
        result["source_files"][split] = dict(filename=filename, sha256=digest(path))
        rows, exclusions = [], []
        for row in pd.read_parquet(path).to_dict("records"):
            texts, labels = list(row["choices"]["text"]), list(row["choices"]["label"])
            if len(texts) != 4 or row["answerKey"] not in labels:
                exclusions.append(row["id"])
                continue
            answer = labels.index(row["answerKey"])
            h = hashlib.sha256(f'{SEED}:{row["id"]}'.encode()).hexdigest()
            wrong = [i for i in range(4) if i != answer][int(h, 16) % 3]
            rows.append(dict(id=row["id"], question=row["question"], choices=texts,
                             answer=answer, wrong=wrong, selection_hash=h))
        rows.sort(key=lambda x: x["selection_hash"])
        result["splits"][split] = dict(rows=rows[:n], available_four_choice=len(rows),
                                       excluded_non_four_choice=exclusions)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(revision=revision, sha256=digest(dest),
                          counts={s: len(v["rows"]) for s, v in result["splits"].items()})))


if __name__ == "__main__":
    main()
