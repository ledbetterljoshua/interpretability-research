"""Download pinned public reference files and hash them; never load a model."""
import os
os.environ.update(HF_HUB_OFFLINE="0",HF_HUB_DISABLE_PROGRESS_BARS="1",HF_HUB_DOWNLOAD_TIMEOUT="60")
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import threading
import time
from huggingface_hub import HfApi,snapshot_download
from runtime import ROOT,sha,atomic_json

PLAN=ROOT/"notes/2026-09-12-causal-audit-unmodified-reference-download-plan.md"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
FILES=["LICENSE","README.md","config.json","generation_config.json","merges.txt",
       "model.safetensors","tokenizer.json","tokenizer_config.json","vocab.json"]


def main():
    committed=subprocess.check_output(["git","show",f"HEAD:{PLAN.relative_to(ROOT)}"],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==sha(PLAN)
    out=ROOT/"data/causal_audit/unmodified-reference-download-v1";out.mkdir(exist_ok=False)
    started=time.monotonic();done=threading.Event();mutex=threading.RLock()
    m=dict(status="running",model=MODEL,revision=REVISION,model_weights_loaded=False,
        elapsed_limit_seconds=1200,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            (PLAN,Path(__file__),Path(__file__).with_name("runtime.py"))},
        huggingface_hub_version=importlib.metadata.version("huggingface_hub"))
    def save(**updates):
        with mutex:
            m.update(updates);m["elapsed_seconds"]=time.monotonic()-started
            atomic_json(out/"download.json",m)
    def watch():
        if not done.wait(1200):
            save(status="resource_stopped",error="Download time cap exceeded")
            os._exit(75)
    save();thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:
        info=HfApi().model_info(MODEL,revision=REVISION,files_metadata=True,token=False)
        assert info.sha==REVISION and info.private is False and info.gated is False
        assert info.card_data.get("license")=="apache-2.0"
        sizes={s.rfilename:s.size for s in info.siblings if s.rfilename in FILES}
        assert set(sizes)==set(FILES) and all(type(s) is int and s>0 for s in sizes.values())
        assert sum(sizes.values())<4*1024**3
        save(stage="downloading",declared_sizes=sizes,license="apache-2.0",public=True,ungated=True)
        snapshot=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=FILES,max_workers=1,token=False))
        hashes={}
        for name,size in sizes.items():
            path=snapshot/name;assert path.stat().st_size==size,name
            hashes[name]=sha(path)
        config=json.loads((snapshot/"config.json").read_text())
        original=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        post=json.loads((original/"config.json").read_text())
        dimensions=("model_type","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads",
                    "num_key_value_heads","head_dim","vocab_size","tie_word_embeddings")
        assert all(config[k]==post[k] for k in dimensions)
        assert config["max_position_embeddings"]>=512
        a=json.loads((snapshot/"tokenizer.json").read_text());b=json.loads((original/"tokenizer.json").read_text())
        same_vocab=a["model"]["vocab"]==b["model"]["vocab"]
        added=lambda t:{r["id"]:r["content"] for r in t["added_tokens"]}
        same_added=added(a)==added(b)
        save(stage="verified",snapshot=str(snapshot),file_hashes=hashes,
            compatible_dimensions={k:config[k] for k in dimensions},same_tokenizer_vocabulary=same_vocab,
            same_added_token_ids=same_added,post_config_sha256=sha(original/"config.json"),
            post_tokenizer_sha256=sha(original/"tokenizer.json"))
        assert same_vocab and same_added,"Tokenizer mapping mismatch"
        save(status="complete")
        print(json.dumps(m,indent=2),flush=True)
    except BaseException as error:
        save(status="error",error=f"{type(error).__name__}: {error}");raise
    finally:done.set();thread.join(timeout=2)


if __name__=="__main__":main()
