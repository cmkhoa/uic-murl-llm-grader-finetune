"""Run one frozen v0.7 arm without rerunning an already-preserved base/v0.5 arm."""
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


family = os.environ["MURL_V08_FAMILY"]
run_label = os.environ["MURL_V08_RUN_LABEL"]
config_path = os.environ["MURL_V08_V07_CONFIG"]
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

from huggingface_hub import snapshot_download
from safetensors import safe_open
import yaml

config = yaml.safe_load(Path(config_path).read_text())
base = config["base_model"]
training = config["training"]
assert training["max_steps"] == 60
assert training["two_stage_diagnosis"]["stage_1"]["steps"] == 30
assert training["two_stage_diagnosis"]["stage_2"]["steps"] == 30
assert training["save_steps"] == 30
assert training.get("save_total_limit", 1) >= 2
local_cache = Path(f"/content/murl-hf-v08-{family}")
local_snapshot = local_cache / ("models--" + base["id"].replace("/", "--")) / "snapshots" / base["revision"]
snapshot_download(base["id"], revision=base["revision"], token=False, local_dir=str(local_snapshot))
for file in local_snapshot.glob("*.safetensors"):
    with safe_open(file, framework="pt", device="cpu") as weights:
        print(file.name, file.stat().st_size, len(weights.keys()), "tensors verified", flush=True)
os.environ["HF_HUB_CACHE"] = str(local_cache)
os.environ["HF_HUB_OFFLINE"] = "1"

assert file_sha256(config["data"]["train_file"]) == config["data"]["train_file_sha256"]
assert file_sha256(config["data"]["validation_file"]) == config["data"]["validation_file_sha256"]
run_logged([sys.executable, "-u", "scripts/run_qlora_smoke.py", "--config", config_path])

model_dir = Path(training["output_dir"])
checkpoint = model_dir / "checkpoint-30"
assert checkpoint.is_dir(), "Required Stage 1 checkpoint was not retained"
shutil.copy2(archive, model_dir / "source.zip")
shutil.copy2("data/training/v0.5-sft-train.metadata.json", model_dir / "dataset_metadata.json")


def preserve(source, name):
    destination = Path("/content") / name
    shutil.copy2(source, destination)
    with zipfile.ZipFile(destination) as saved:
        assert saved.testzip() is None
    drive_destination = backup / name
    shutil.copy2(destination, drive_destination)
    local_hash = file_sha256(destination)
    drive_hash = file_sha256(drive_destination)
    assert local_hash == drive_hash
    print(name, local_hash, "local/Drive match", flush=True)


training_zip = shutil.make_archive(f"/content/{family}-v08-v07-training", "zip", root_dir=root, base_dir=model_dir)
preserve(training_zip, f"{family}-v08-v07-training-results.zip")
shutil.copytree(model_dir / "adapter", backup / "v07-adapter", dirs_exist_ok=True)
shutil.copytree(checkpoint, backup / "v07-stage-1-checkpoint", dirs_exist_ok=True)
run_logged([
    sys.executable, "-u", "scripts/evaluate_qlora_adapter.py", "--config", config_path,
    "--adapter", str(model_dir / "adapter"),
])
preserve(
    root / "runs" / (config["adapter_inference"]["run_id"] + ".zip"),
    f"{family}-v08-v07-validation-results.zip",
)
print("EXPERIMENT COMPLETE: frozen v0.7 artifacts verified and saved to Drive.", flush=True)
