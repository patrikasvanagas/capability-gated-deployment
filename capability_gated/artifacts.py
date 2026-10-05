"""Run-directory layout, JSON/figure writers, and provenance capture.

Every mechanistic-interpretability or search run must leave behind a
self-describing artifact directory: a pre-registration record written *before*
any data is touched (the statistical protocol's rule that adaptive choices are
frozen up front), the measured results, the figures, and a manifest. This module
owns that bookkeeping so the exploratory notebook and the later, headless
configuration-search run write byte-compatible records instead of two dialects.

``jsonable`` / ``write_json`` / ``save_figure`` are ports of the helpers in
``scripts/finetune_and_gate.py`` -- they live here so package code and scripts
can share them (the script predates this module and keeps its own copies).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Self, Sequence

import numpy as np
import torch


def jsonable(obj: Any) -> Any:
    """Recursively unwrap numpy/torch scalars so ``json.dumps`` accepts them."""
    if isinstance(obj, dict):
        out: Any = {str(k): jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        out = [jsonable(v) for v in obj]
    elif isinstance(obj, np.integer):
        out = int(obj)
    elif isinstance(obj, np.floating):
        out = float(obj)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        out = obj.tolist()
    elif isinstance(obj, Path):
        out = str(obj)
    else:
        out = obj
    return out


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")
    return path


def save_figure(
    fig: Any,
    stem: str | Path,
    data: Any = None,
) -> list[Path]:
    """Write a figure as PNG (quick look) and PDF (paper), then close it.

    Closing matters in notebooks: a long analysis session that keeps every
    figure open accumulates hundreds of MB of render buffers.

    ``data`` is the series actually plotted. When given it is written beside the
    image as ``<stem>.data.json``, so a figure can be rebuilt in any tool ---
    the paper's own pgfplots/matplotlib style, say --- without reading the
    plotting code to work out what it derived. A figure whose numbers exist only
    inside a PDF is a figure that has to be regenerated to be reused.
    """
    import matplotlib.pyplot as plt  # heavy; keep the module import cheap

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        target = stem.with_suffix(f".{ext}")
        fig.savefig(target, dpi=300, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    if data is not None:
        written.append(write_json(stem.with_suffix(".data.json"), data))
    print(f"wrote {' / '.join(p.name for p in written)} -> {stem.parent}")
    return written


class RunLog:
    """Mirror stdout and stderr into ``<run>/run.log`` while still printing.

    The blanket safety net. Every artifact schema in this package is a
    deliberate choice about what to keep, and every such choice can be wrong;
    the transcript catches whatever the schemas missed -- progress lines, warning
    text, the exact wording of an assertion that fired -- at the cost of a few
    hundred kilobytes. Use as a context manager so the streams are restored even
    when the run raises.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: Any = None
        self._stdout: Any = None
        self._stderr: Any = None

    class _Tee:
        def __init__(self, stream: Any, sink: Any) -> None:
            self.stream, self.sink = stream, sink

        def write(self, text: str) -> int:
            self.stream.write(text)
            self.sink.write(text)
            self.sink.flush()
            return len(text)

        def flush(self) -> None:
            self.stream.flush()
            self.sink.flush()

        def isatty(self) -> bool:
            return bool(getattr(self.stream, "isatty", lambda: False)())

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._file.write(
            f"\n===== run started {datetime.now(tz=UTC).isoformat(timespec='seconds')} "
            f"=====\n$ {' '.join(sys.argv)}\n",
        )
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._Tee(self._stdout, self._file)  # type: ignore[assignment]
        sys.stderr = self._Tee(self._stderr, self._file)  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stdout is not None:
            sys.stdout, sys.stderr = self._stdout, self._stderr
        if self._file is not None:
            self._file.close()
            self._file = None


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (checkpoints are GBs; never read them whole)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Run a git command, or return None if it cannot be answered.

    None means "unknown", never "no" -- callers must not coerce it to a boolean
    without deciding what an unanswerable question should record.

    The timeout is generous because `status --porcelain` walks the working tree,
    and on a pod that has just written a multi-GB checkpoint into an ignored
    directory that walk is disk-bound: measured at 32.5s on an L40S pod against
    the old 10s limit. TimeoutExpired is a SubprocessError, NOT an OSError, so
    the old handler let it escape and kill the caller -- which is how a 23-hour
    cyber search died during --estimate-only before running a single evaluation.
    """
    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed executable, literal args
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Capture everything needed to say later *which code produced this*.

    A dirty working tree is recorded rather than blocked -- but a
    pre-registration record that points at uncommitted code is not much of a
    record, so callers should print a loud warning when ``git.dirty`` is true.

    ``git.dirty`` is tri-state: True, False, or None when git could not answer.
    None must not be read as clean -- an artifact that silently claims a clean
    tree it never verified is worse than one that admits it does not know.
    """
    import datasets
    import transformers

    status = _git("status", "--porcelain")

    record: dict[str, Any] = {
        "created_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        # The command is part of the provenance: several settings reach a run
        # only through the CLI, and reconstructing them from the artifacts
        # afterwards is guesswork.
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
        },
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
        record["cuda"] = torch.version.cuda
    if extra:
        record.update(dict(extra))
    return record


@dataclass(frozen=True)
class RunPaths:
    """The artifact directory of one experiment run, with fixed sub-layout.

    ``create`` builds ``<base>/<experiment>/<checkpoint_name>/<stamp>[_smoke]``.
    The ``_smoke`` suffix is not cosmetic: a capped debug run must never be able
    to overwrite a real run's artifacts, and a distinct directory name is the
    simplest guarantee.
    """

    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def attribution(self) -> Path:
        return self.root / "attribution"

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    @classmethod
    def create(
        cls,
        base: str | Path,
        *,
        checkpoint_name: str,
        experiment: str = "e2_localisation",
        stamp: str | None = None,
        smoke: bool = False,
    ) -> RunPaths:
        if stamp is None:
            stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        if smoke:
            stamp += "_smoke"
        root = Path(base) / experiment / checkpoint_name / stamp
        for sub in (root, root / "figures", root / "attribution", root / "validation"):
            sub.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def write_readme(self, descriptions: Mapping[str, str], header: str = "") -> Path:
        """Write a README.md describing what each file in this run dir holds.

        An artifact directory that cannot explain itself is one you have to read
        the source to use, and the source moves. ``descriptions`` maps a path (or
        a glob-ish prefix) to one line of prose; anything present on disk but
        undescribed is still listed, flagged, so the README cannot silently fall
        behind the code that writes the files.
        """
        present = sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and p.name != "README.md"
        )
        lines = [f"# {self.root.name}", ""]
        if header:
            lines += [header, ""]
        lines += ["| file | bytes | what it holds |", "|---|---:|---|"]

        def describe(rel: str) -> str:
            if rel in descriptions:
                return descriptions[rel]
            for key, text in descriptions.items():
                if key.endswith("/") and rel.startswith(key):
                    return text
                if key.endswith("*") and rel.startswith(key[:-1]):
                    return text
            return "**(undescribed --- update `descriptions`)**"

        for rel in present:
            size = (self.root / rel).stat().st_size
            lines.append(f"| `{rel}` | {size:,} | {describe(rel)} |")
        lines.append("")
        path = self.root / "README.md"
        path.write_text("\n".join(lines))
        return path

    def manifest(self) -> dict[str, Any]:
        """Every file under the run dir with size and sha256, for the record."""
        files = [
            {
                "path": str(path.relative_to(self.root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
        return {"root": str(self.root), "n_files": len(files), "files": files}


def format_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    float_fmt: str = "{:.4f}",
) -> str:
    """Plain-text aligned table for notebook printing.

    Exists so package output does not depend on pandas, which is installed only
    transitively and deliberately undeclared in ``pyproject.toml``.
    """
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return float_fmt.format(float(value))
        return str(value)

    table = [[render(row.get(col, "")) for col in columns] for row in rows]
    widths = [
        max(len(col), *(len(line[i]) for line in table)) if table else len(col)
        for i, col in enumerate(columns)
    ]
    header = "  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    rule = "  ".join("-" * w for w in widths)
    body = ("  ".join(line[i].ljust(widths[i]) for i in range(len(columns))) for line in table)
    return "\n".join([header, rule, *body])
