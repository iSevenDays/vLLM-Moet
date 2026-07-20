#!/usr/bin/env python3
"""Catch use-before-local-import scope bugs in the patched vllm files.

The bug class (found the expensive way, at first model construction on
the v0.25.1 lineage): a function uses a module-level-imported name, and
a LATER line in the SAME function re-imports that name locally — Python
then treats the name as function-local everywhere in the scope, so the
earlier use dies with UnboundLocalError at runtime. Merges create this
silently: two sides' blocks land at different anchors of one function,
each self-consistent, their union broken (sparse_attn_indexer.__init__
and mla/indexer.__init__ both had it after the 0.25.1 port merge).

py_compile cannot see it; flake8's F823 is routinely noqa'd upstream.
This checker walks every function in every file a lineage patch
touches and flags any Name LOAD that precedes a local import binding
of the same name in that scope. Names inside annotations are skipped
(function-body annotations are never evaluated at runtime — upstream's
noqa'd `self.drafter: NgramProposer | ...` pattern is legal).

Run from the repo root (after the fork worktree is checked out at the
lineage branch, e.g. inside the fork clone):

    python3 tools/check_scope_shadowing.py [overlay-python-file ...]

defaults to every overlay/vllm/**/*.py file (the source of truth since
the overlay-patch workflow refactor); file paths resolve against the
CURRENT working directory (run it from the vllm tree being checked).
Exit 1 on findings.
"""
import ast
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def annotation_names(tree) -> set[int]:
    """id()s of Name nodes that live inside annotation expressions."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        anns = []
        if isinstance(node, ast.AnnAssign):
            anns.append(node.annotation)
        elif isinstance(node, ast.arg) and node.annotation is not None:
            anns.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.returns is not None:
            anns.append(node.returns)
        for a in anns:
            for sub in ast.walk(a):
                if isinstance(sub, ast.Name):
                    skip.add(id(sub))
    return skip


def check_file(fn: str) -> int:
    try:
        tree = ast.parse(open(fn).read(), fn)
    except SyntaxError as e:
        print(f"SYNTAX {fn}: {e}")
        return 1
    except OSError:
        return 0            # file absent in this tree (other lineage)
    skip = annotation_names(tree)
    bad = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local_imports: dict[str, int] = {}
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    nm = a.asname or a.name.split(".")[0]
                    local_imports[nm] = min(
                        local_imports.get(nm, 1 << 30), sub.lineno)
        if not local_imports:
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
                    and id(sub) not in skip
                    and sub.id in local_imports
                    and sub.lineno < local_imports[sub.id]):
                print(f"USE-BEFORE-LOCAL-IMPORT {fn}:{sub.lineno} "
                      f"'{sub.id}' becomes local at line "
                      f"{local_imports[sub.id]} (function {node.name}) "
                      "-> UnboundLocalError at runtime")
                bad += 1
    return bad


def main() -> int:
    if sys.argv[1:]:
        files = {a for a in sys.argv[1:] if a.endswith(".py")}
    else:
        overlay_root = os.path.join(REPO, "overlay", "vllm")
        files = {
            os.path.relpath(os.path.join(dp, f), REPO)
            for dp, _ds, fs in os.walk(overlay_root)
            for f in fs if f.endswith(".py")
        }
    bad = 0
    checked = 0
    for fn in sorted(files):
        if os.path.exists(fn):
            checked += 1
            bad += check_file(fn)
    print(f"scope-shadowing check: {checked} files, {bad} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
