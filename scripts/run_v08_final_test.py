#!/usr/bin/env python3
"""Run the single authorized MURL v0.8 held-out test on Qwen3 v0.6.

This file is intended for a fresh CUDA Colab runtime. It is deliberately
fail-closed: frozen artifacts and the model are verified before the immutable
run directory is created and before the test JSONL is parsed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

RUN_ID = "qwen3-4b-murl-v0.8-v0.6-final-test-20261103"
SOURCE_ARCHIVE = Path("/content/murl-v08-v06-cross-family-source.zip")
TRAINING_ARCHIVE = Path("/content/qwen3-4b-v08-v06-training-results.zip")
EXPECTED = {
    "source_archive": "f4d6e82d14b9fad9465263ac3528d50aa0968c0c4e6e033c210ce549816615c1",
    "training_archive": "35ac9d8293df2ddc85c051fca14eb8da7e59540b3ce6277972839cb13c190610",
    "test": "b3f61a20f95c2050a9bf8d8ef01cc3ff5d56c9996888d404a83d351f563bdf59",
    "prompt": "bba7ac0e231d661b7578dab544cc6611f65382889ac8632c9efcd086f37ed2b6",
    "evaluator": "e3b84c6953297af9e7a8d46c5c15616c282dd6908d2488a31488072a644cbd6c",
    "prepare_run": "fe736e402796ce0593a1eda4647679264f9ed7382a7af8cd2f41291c24517006",
    "split_manifest": "40cfd584a0f5176fa71d70df2c8be32db34ebcea9dc8dd7222662577a84a19f2",
    "raw_dataset": "80ed4a372590baf975424e932aa4718b64ef6b7c0e54450a3e74e63b83686f4d",
    "resolved_config": "ec59b1e44176e4b96eb2afd491fbb489dd66725c19e84068c65c4b686a2c68d7",
    "adapter_config": "ac7a01d415e2a6fc808d293b6faf4af25c3d7fa97900bcfa63eca470ca931048",
    "tokenizer_config": "c4f4c62b741ab2940841480f9942f378d7d3878d47c2babb862af7381c59b952",
    "chat_template": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
}
BASE_ID = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_PREFIX = "models/qwen3-4b-murl-v0.8-v0.6/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"hash mismatch for {path}: {actual} != {expected}")


def member_hash(archive: zipfile.ZipFile, suffix: str) -> tuple[str, str]:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise AssertionError(f"expected one {suffix}, got {names}")
    return names[0], hashlib.sha256(archive.read(names[0])).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    # Byte-level preflight only: no held-out examples are parsed here.
    if not TRAINING_ARCHIVE.exists():
        parts = sorted(Path("/content").glob(TRAINING_ARCHIVE.name + ".part-*"))
        if not parts:
            raise FileNotFoundError(f"missing {TRAINING_ARCHIVE} and upload chunks")
        with TRAINING_ARCHIVE.open("xb") as destination:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination)
        print(f"REASSEMBLED_TRAINING_ARCHIVE_FROM_{len(parts)}_PARTS", flush=True)
    assert SOURCE_ARCHIVE.is_file() and TRAINING_ARCHIVE.is_file()
    assert_hash(SOURCE_ARCHIVE, EXPECTED["source_archive"])
    assert_hash(TRAINING_ARCHIVE, EXPECTED["training_archive"])
    with zipfile.ZipFile(SOURCE_ARCHIVE) as source_zip:
        assert source_zip.testzip() is None
    with zipfile.ZipFile(TRAINING_ARCHIVE) as training_zip:
        assert training_zip.testzip() is None
        frozen_members = {}
        for label, suffix in (
            ("resolved_config", "/resolved_config.json"),
            ("adapter_config", "/adapter/adapter_config.json"),
            ("tokenizer_config", "/tokenizer/tokenizer_config.json"),
            ("chat_template", "/tokenizer/chat_template.jinja"),
        ):
            name, digest = member_hash(training_zip, suffix)
            if digest != EXPECTED[label]:
                raise AssertionError(f"{label} member hash mismatch")
            frozen_members[label] = {"member": name, "sha256": digest}
        adapter_names = [name for name in training_zip.namelist() if name.endswith("/adapter/adapter_model.safetensors")]
        if len(adapter_names) != 1:
            raise AssertionError("missing unique final adapter weights")
        frozen_members["adapter_weights"] = {
            "member": adapter_names[0],
            "sha256": hashlib.sha256(training_zip.read(adapter_names[0])).hexdigest(),
            "bytes": training_zip.getinfo(adapter_names[0]).file_size,
        }

    existing_runs = list(Path("/content").glob(f"murl-v08-final-source-*/runs/{RUN_ID}"))
    if len(existing_runs) > 1:
        raise RuntimeError(f"ambiguous immutable run directories: {existing_runs}")
    resume_setup_failure = len(existing_runs) == 1
    if resume_setup_failure:
        work_root = existing_runs[0].parents[1]
        print("STRICT_RESUME_OF_PRE_GENERATION_SETUP_FAILURE", work_root, flush=True)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="murl-v08-final-source-", dir="/content"))
        with zipfile.ZipFile(SOURCE_ARCHIVE) as source_zip:
            source_zip.extractall(work_root)
        with zipfile.ZipFile(TRAINING_ARCHIVE) as training_zip:
            selected = [
                name for name in training_zip.namelist()
                if name.startswith(MODEL_PREFIX + "adapter/") or name.startswith(MODEL_PREFIX + "tokenizer/")
                or name in {MODEL_PREFIX + "metadata.json", MODEL_PREFIX + "resolved_config.json", MODEL_PREFIX + "dataset_metadata.json"}
            ]
            for name in selected:
                training_zip.extract(name, work_root)

    test_path = work_root / "data/processed/test.jsonl"
    prompt_path = work_root / "prompts/v0.2-sft-system.md"
    evaluator_path = work_root / "evaluation/evaluate.py"
    assert_hash(test_path, EXPECTED["test"])
    assert_hash(prompt_path, EXPECTED["prompt"])
    assert_hash(evaluator_path, EXPECTED["evaluator"])
    assert_hash(work_root / "scripts/prepare_run.py", EXPECTED["prepare_run"])
    assert_hash(work_root / "data/splits/v0.1.json", EXPECTED["split_manifest"])
    assert_hash(work_root / "data/raw/graded-exams.jsonl", EXPECTED["raw_dataset"])
    # Count newline-delimited records without decoding or parsing examples/labels.
    if sum(1 for line in test_path.open("rb") if line.strip()) != 47:
        raise AssertionError("held-out split count mismatch")

    config = json.loads((work_root / (MODEL_PREFIX + "resolved_config.json")).read_text())
    base = config["base_model"]
    assert base["id"] == BASE_ID and base["revision"] == BASE_REVISION
    assert base["chat_template_kwargs"] == {"enable_thinking": False}
    assert base["quantization"] == "nf4" and base["compute_dtype"] == "float16"
    assert config["adapter_inference"]["system_prompt_file"] == "prompts/v0.2-sft-system.md"
    assert config["adapter_inference"]["max_new_tokens"] == 256

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("authorized final test requires CUDA")
    set_seed(config["run"]["seed"])
    tokenizer_dir = work_root / (MODEL_PREFIX + "tokenizer")
    adapter_dir = work_root / (MODEL_PREFIX + "adapter")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=BASE_REVISION, token=False, device_map="auto",
        quantization_config=quantization, torch_dtype=torch.float16, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("MODEL_AND_ADAPTER_LOAD_VERIFIED", flush=True)

    # Immutable one-time boundary. Nothing above parsed a held-out record.
    run_dir = work_root / "runs" / RUN_ID
    results_dir = work_root / "results" / RUN_ID
    raw_archive = Path("/content") / f"{RUN_ID}-raw-pre-score.zip"
    final_archive = Path("/content") / f"{RUN_ID}.zip"
    if resume_setup_failure:
        expected_existing = {"metadata.json", "prompt.md", "decoder.json", "frozen-inputs.json", SOURCE_ARCHIVE.name, "gold.jsonl"}
        actual_existing = {path.name for path in run_dir.iterdir()}
        if actual_existing != expected_existing or results_dir.exists() or raw_archive.exists() or final_archive.exists():
            raise FileExistsError(f"run is not in the exact zero-output resumable state: {actual_existing}")
        resume_metadata = json.loads((run_dir / "metadata.json").read_text())
        resume_metadata["setup_failures"] = [{
            "stage": "after held-out parse, before tokenization/generation",
            "error": "ModuleNotFoundError: scripts",
            "outputs_generated": 0,
            "recovery": "same immutable run; original test not reread; generation uses frozen gold copy",
        }]
        (run_dir / "metadata.json").write_text(json.dumps(resume_metadata, indent=2, sort_keys=True) + "\n")
    else:
        for path in (run_dir, results_dir, raw_archive, final_archive):
            if path.exists():
                raise FileExistsError(f"immutable final-test target already exists: {path}")
        run_dir.mkdir(parents=True, exist_ok=False)

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    runtime = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "authorization": "single frozen held-out evaluation of selected Qwen3 v0.6 winner",
        "model_id": BASE_ID,
        "model_revision": BASE_REVISION,
        "adapter_run_id": "qwen3-4b-murl-v0.8-v0.6",
        "seed": config["run"]["seed"],
        "gpu": torch.cuda.get_device_name(0),
        "packages": {name: version(name) for name in ["torch", "transformers", "accelerate", "bitsandbytes", "peft"]},
        "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "torch.float16"},
        "chat_template_kwargs": {"enable_thinking": False},
        "decoding": {"do_sample": False, "max_new_tokens": 256, "retries": 0, "repair": False},
        "input_sha256": {**EXPECTED, **{f"training_member_{k}": v["sha256"] for k, v in frozen_members.items()}},
        "test_examples": 47,
        "test_access_before_generation": "hash and byte-line count only",
    }
    if not resume_setup_failure:
        (run_dir / "metadata.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
        (run_dir / "decoder.json").write_text(json.dumps(runtime["decoding"], indent=2) + "\n", encoding="utf-8")
        (run_dir / "frozen-inputs.json").write_text(json.dumps(frozen_members, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(SOURCE_ARCHIVE, run_dir / SOURCE_ARCHIVE.name)

    sys.path.insert(0, str(work_root))
    from scripts.make_sft_dataset import user_message
    # First semantic held-out read for a fresh run. A strict setup continuation
    # reads only the already-frozen gold copy, never the original test file.
    semantic_input = run_dir / "gold.jsonl" if resume_setup_failure else test_path
    examples = [json.loads(line) for line in semantic_input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(examples) != 47:
        raise AssertionError("held-out parse count mismatch")
    gold_path = run_dir / "gold.jsonl"
    if not resume_setup_failure:
        shutil.copy2(test_path, gold_path)
    outputs_path = run_dir / "outputs.jsonl"
    with outputs_path.open("x", encoding="utf-8") as handle:
        for index, example in enumerate(examples, 1):
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message(example)},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            handle.write(json.dumps({"example_id": example["example_id"], "output": text}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"GENERATED {index}/47", flush=True)

    # Freeze raw outputs before the sole scoring invocation.
    with zipfile.ZipFile(raw_archive, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in (run_dir / "metadata.json", run_dir / "prompt.md", run_dir / "decoder.json",
                     run_dir / "frozen-inputs.json", gold_path, outputs_path):
            zf.write(path, arcname=path.name)
    with zipfile.ZipFile(raw_archive) as zf:
        assert zf.testzip() is None
    raw_hash = sha256(raw_archive)
    shutil.copy2(raw_archive, run_dir / raw_archive.name)
    print("RAW_OUTPUTS_ARCHIVED_BEFORE_SCORING", raw_hash, flush=True)

    subprocess.run([
        sys.executable, str(evaluator_path), "--gold", str(gold_path), "--outputs", str(outputs_path),
        "--run-id", RUN_ID, "--results-dir", str(work_root / "results"),
    ], check=True, cwd=work_root)
    shutil.copy2(run_dir / "metadata.json", results_dir / "metadata.json")
    shutil.copytree(results_dir, run_dir / "results")

    per_example = [json.loads(line) for line in (results_dir / "per-example.jsonl").read_text().splitlines() if line.strip()]
    valid = [row for row in per_example if row["output_valid"]]
    labels = sorted({row["gold_error_taxonomy"] for row in per_example})
    confusions = Counter((row["gold_error_taxonomy"], row["parsed_output"]["error_taxonomy"]) for row in valid)
    write_csv(run_dir / "taxonomy-confusion.csv", [
        {"gold": gold, "predicted": pred, "count": confusions[(gold, pred)]}
        for gold in labels for pred in labels
    ])
    slice_rows = []
    for dimension in ("topic", "difficulty"):
        groups = defaultdict(list)
        for row in per_example:
            groups[row[dimension]].append(row)
        for value, rows in sorted(groups.items()):
            vrows = [row for row in rows if row["output_valid"]]
            localized = [row for row in vrows if isinstance(row["gold_first_error_step"], int)]
            slice_rows.append({
                "dimension": dimension, "slice": value, "examples": len(rows), "valid": len(vrows),
                "correct_exact": sum(bool(row.get("correctness_exact")) for row in vrows),
                "taxonomy_exact": sum(bool(row.get("taxonomy_exact")) for row in vrows),
                "first_error_support": len(localized),
                "first_error_exact": sum(bool(row.get("first_error_exact")) for row in localized),
            })
    write_csv(run_dir / "slices.csv", slice_rows)
    manifest = {
        "raw_pre_score_zip": {"file": raw_archive.name, "sha256": raw_hash},
        "files": {
            str(path.relative_to(run_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(run_dir.rglob("*")) if path.is_file()
        },
        "scoring_invocations": 1,
        "generation_passes": 1,
    }
    (run_dir / "sha256-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(final_archive, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(RUN_ID) / path.relative_to(run_dir)))
    with zipfile.ZipFile(final_archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise AssertionError(f"final ZIP CRC failure: {bad}")
    final_hash = sha256(final_archive)
    print("FINAL_TEST_COMPLETE", RUN_ID, final_hash, flush=True)
    print(json.dumps(json.loads((results_dir / "aggregate.json").read_text()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
