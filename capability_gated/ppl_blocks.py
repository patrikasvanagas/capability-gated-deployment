# ruff: noqa: CPY001
"""A once-built, fingerprinted cache of the held-out perplexity blocks.

Every evaluation pod used to tokenize the full MiniPile -- ~40 minutes, ~38 of
which processed the 1M-document train split that perplexity never reads. This
module makes the blocks a shared artifact instead: the first run builds the
validation blocks (~2 min), uploads them to a HF dataset repo, and every later
run downloads them in seconds.

The cache is **self-populating and fail-open**: no artifact -> build locally
(and upload when a token is available); download or verification trouble ->
build locally. A run never fails because the cache is unreachable.

The **fingerprint** is what makes reuse safe rather than convenient: the blocks
are only valid for the exact tokenizer, dataset, split and block size that
produced them, so the artifact name and a stored manifest both carry a hash of
those. A mismatch -- say a substrate with a different vocabulary -- falls back
to building rather than silently scoring on foreign blocks. It also upgrades a
convention into a guarantee: every run that loads the same fingerprint scores
perplexity on byte-identical blocks.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from .training import prepare_lm_dataset

MANIFEST = "fingerprint.json"


def _tokenizer_hash(tokenizer: Any) -> str:
    """Hash of the vocabulary, the part of the tokenizer that changes blocks."""
    vocab = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(vocab.encode()).hexdigest()[:16]


def blocks_fingerprint(
    tokenizer: Any, *, dataset: str, dataset_config: str | None,
    split: str, block_size: int,
) -> dict[str, Any]:
    ident = {
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "block_size": int(block_size),
        "tokenizer_sha": _tokenizer_hash(tokenizer),
    }
    digest = hashlib.sha256(
        json.dumps(ident, sort_keys=True).encode(),
    ).hexdigest()[:12]
    return {**ident, "digest": digest}


def load_or_build_ppl_blocks(
    tokenizer: Any,
    *,
    dataset: str = "JeanKaddour/minipile",
    dataset_config: str | None = None,
    split: str = "validation",
    block_size: int = 256,
    num_workers: int = 8,
    cache_dir: str | None = None,
    repo_id: str | None = None,
    verbose: bool = True,
) -> Dataset:
    """The perplexity blocks: downloaded when cached, built (and cached) if not.

    ``repo_id`` defaults to ``$PPL_BLOCKS_REPO``, then
    ``$HF_NAMESPACE/ppl-blocks``. With neither set (or ``PPL_BLOCKS_REPO=off``)
    the blocks are always built locally and nothing touches the Hub.
    """
    fp = blocks_fingerprint(
        tokenizer, dataset=dataset, dataset_config=dataset_config,
        split=split, block_size=block_size,
    )
    name = f"{dataset.split('/')[-1]}-{split}-bs{block_size}-{fp['digest']}"
    namespace = os.environ.get("HF_NAMESPACE")
    repo_id = repo_id or os.environ.get("PPL_BLOCKS_REPO") or (
        f"{namespace}/ppl-blocks" if namespace else None
    )
    log = print if verbose else (lambda *_: None)

    if repo_id and repo_id != "off":
        try:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(
                repo_id=repo_id, repo_type="dataset",
                allow_patterns=[f"{name}/*"], cache_dir=cache_dir,
            )
            root = Path(snap) / name
            stored = json.loads((root / MANIFEST).read_text())
            if stored != fp:
                raise ValueError(
                    f"fingerprint mismatch: artifact {stored['digest']} vs "
                    f"expected {fp['digest']}",
                )
            ds = load_from_disk(str(root / "blocks"))
            log(f"ppl blocks: reused {repo_id}/{name} ({len(ds)} blocks)")
            return ds
        except Exception as exc:  # noqa: BLE001 - fail-open by design
            log(f"ppl blocks: no reusable artifact ({type(exc).__name__}: {exc}); building")

    ds = prepare_lm_dataset(
        dataset, tokenizer, block_size=block_size, num_workers=num_workers,
        dataset_config=dataset_config, dataset_split=split, cache_dir=cache_dir,
    )
    log(f"ppl blocks: built {len(ds)} blocks from {dataset}:{split}")

    if repo_id and repo_id != "off" and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / name
                ds.save_to_disk(str(root / "blocks"))
                (root / MANIFEST).write_text(json.dumps(fp, indent=2))
                api = HfApi()
                api.create_repo(repo_id, repo_type="dataset", private=True,
                                exist_ok=True)
                api.upload_folder(repo_id=repo_id, repo_type="dataset",
                                  folder_path=str(root), path_in_repo=name)
            log(f"ppl blocks: uploaded {repo_id}/{name} for future runs")
        except Exception as exc:  # noqa: BLE001 - the run itself must not fail
            log(f"ppl blocks: upload skipped ({type(exc).__name__}: {exc})")
    return ds
