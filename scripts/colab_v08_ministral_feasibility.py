"""Run the required non-scored T4 feasibility gate for Ministral 3 3B.

This intentionally performs no validation. It verifies a frozen-setting base
load, QLoRA attachment, one optimizer step, and saved-adapter reload before
Ministral may enter any scored comparison.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def run_logged(args):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, args)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


family = "ministral3-3b"
run_label = os.environ.get("MURL_V08_RUN_LABEL", "ministral3-3b-t4-smoke-20261101")
config_path = os.environ.get(
    "MURL_V08_MINISTRAL_SMOKE_CONFIG",
    "configs/qlora-v0.8-ministral3-v0.5-t4-smoke.yaml",
)
drive = Path("/content/drive/MyDrive/MURL")
assert drive.parent.is_dir(), "Mount Drive before running"
os.environ["HF_HOME"] = str(drive / "huggingface-cache")
archive = Path("/content/murl-v08-source.zip")
root = Path(f"/content/UIC-MURL-v08-{run_label}")
assert not root.exists(), "Use a fresh source directory"
with zipfile.ZipFile(archive) as source:
    assert source.testzip() is None
    source.extractall(root)

backup = drive / "runs/v0.8-cross-family" / run_label
backup.mkdir(parents=True, exist_ok=True)
source_backup = backup / archive.name
shutil.copy2(archive, source_backup)
assert file_sha256(archive) == file_sha256(source_backup)
print("source", file_sha256(archive), flush=True)
os.chdir(root)

run_logged([
    sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.1",
    "accelerate==1.15.0", "bitsandbytes==0.50.2", "peft==0.21.2", "pyyaml",
])

import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open

config = yaml.safe_load(Path(config_path).read_text())
assert config["training"]["smoke_only"] is True
assert config["training"]["max_steps"] == 1
base = config["base_model"]
local_cache = Path(f"/content/murl-hf-v08-{family}")
local_snapshot = local_cache / ("models--" + base["id"].replace("/", "--")) / "snapshots" / base["revision"]
snapshot_download(base["id"], revision=base["revision"], token=False, local_dir=str(local_snapshot))
for file in local_snapshot.glob("*.safetensors"):
    with safe_open(file, framework="pt", device="cpu") as weights:
        print(file.name, file.stat().st_size, len(weights.keys()), "tensors verified", flush=True)
os.environ["HF_HUB_CACHE"] = str(local_cache)
os.environ["HF_HUB_OFFLINE"] = "1"

run_logged([sys.executable, "-u", "scripts/run_qlora_smoke.py", "--config", config_path])
model_dir = Path(config["training"]["output_dir"])
shutil.copy2(archive, model_dir / "source.zip")
shutil.copy2("data/training/v0.5-sft-train.metadata.json", model_dir / "dataset_metadata.json")
result = Path(shutil.make_archive(f"/content/{family}-v08-t4-feasibility-smoke", "zip", root_dir=root, base_dir=model_dir))
with zipfile.ZipFile(result) as saved:
    assert saved.testzip() is None
drive_result = backup / f"{family}-v08-t4-feasibility-smoke-results.zip"
shutil.copy2(result, drive_result)
assert file_sha256(result) == file_sha256(drive_result)
print(drive_result.name, file_sha256(result), "local/Drive match", flush=True)
print("FEASIBILITY COMPLETE: base load, adapter attachment, one step, and adapter reload verified.", flush=True)
