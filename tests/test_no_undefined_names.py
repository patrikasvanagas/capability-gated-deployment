# ruff: noqa: ANN001, ANN201, CPY001, INP001, S101
"""Static guard against reading a name that no longer exists in scope.

`py_compile` accepts a stale reference happily -- a NameError only fires when
that line executes, and in this codebase the expensive scripts touch their
record-writing code *last*: after E2, model load, corpus prep and the baseline.
A refactor that leaves one dangling name therefore fails four hours and one GPU
pod later. That is not hypothetical: a refactor that replaced the dataset loader
in find_suppression_configs.py left `minipile` referenced in the ppl.json
record, which crashed a control run at the ~4.5h mark.

The scan is deliberately scope-aware, including *enclosing* function scopes:
a first attempt that only checked module-level plus local names produced false
positives on every closure (e.g. `_ppl_ratio_fn` reading its parent's `args`),
and a scan people learn to ignore is worse than no scan.

Names are resolved conservatively -- anything assigned anywhere in a scope
counts as bound for that whole scope, ignoring execution order -- so this
catches deletions and typos, not use-before-assignment.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TARGETS = [
    "scripts/find_suppression_configs.py",
    "scripts/evaluate_lattice_composition.py",
    "scripts/offdiagonal_sweep.py",
    "scripts/lattice_family_bh.py",
    "capability_gated/ppl_blocks.py",
    "capability_gated/lattice.py",
]
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def _bound_in(node: ast.AST) -> set[str]:
    """Every name this scope binds, not descending into nested function bodies."""
    bound: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        bound |= {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]}
        bound |= {x.arg for x in (a.vararg, a.kwarg) if x}
    body = node.body if isinstance(node.body, list) else [node.body]
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)          # the definition binds; its body is its own scope
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            bound |= set(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return bound


def _loads_in(node: ast.AST) -> list[ast.Name]:
    """Names read in this scope, not descending into nested function bodies."""
    out: list[ast.Name] = []
    body = node.body if isinstance(node.body, list) else [node.body]
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def undefined_names(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text())
    problems: list[tuple[int, str]] = []

    def walk(node: ast.AST, enclosing: set[str]) -> None:
        scope = enclosing | _bound_in(node)
        for name in _loads_in(node):
            if name.id not in scope and name.id not in BUILTINS:
                problems.append((name.lineno, name.id))
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # only direct children of this scope; ast.walk would revisit
                if child is not node and _is_direct_child(node, child):
                    walk(child, scope)

    def _is_direct_child(parent: ast.AST, target: ast.AST) -> bool:
        stack = list(parent.body if isinstance(parent.body, list) else [parent.body])
        while stack:
            n = stack.pop()
            if n is target:
                return True
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.extend(ast.iter_child_nodes(n))
        return False

    walk(tree, set())
    return sorted(set(problems))


@pytest.mark.parametrize("rel", TARGETS)
def test_no_undefined_names(rel):
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    found = undefined_names(path)
    assert not found, (
        f"{rel} reads names that are not bound in scope: "
        + ", ".join(f"line {ln}: {nm}" for ln, nm in found)
        + " -- a stale reference like this only raises when that line runs, "
          "which in these scripts is hours into a GPU run."
    )
