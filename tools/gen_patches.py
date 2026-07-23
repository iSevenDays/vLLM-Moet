#!/usr/bin/env python3
"""Generate per-file unified-diff patches from overlay/ against the vllm/ baseline.

Source of truth: overlay/vllm/<path> (full files, edited directly).
Read-only baseline: vllm/<path> at the v0.25.1 tag.
Output: one unified-diff patch per file under patches/.

Three modes:
  default   walk overlay/vllm/, diff each file against vllm/<same-path>,
            write patches/vllm-<dashed-path>.patch. Remove patches whose
            overlay source is gone. Rewrite patches/MANIFEST.txt and
            patches/SOURCE.txt atomically.
  --verify  for each patch, re-extract the modified file body, compute its
            git blob SHA, and assert equality with the overlay file's git
            blob SHA. Exit non-zero on any mismatch.
  --check   for each patch, run `git -C vllm apply --check` against the
            v0.25.1 baseline. Exit non-zero on any failure.

Filename scheme (R7): patches/vllm-<path-with-slashes-replaced-by-dashes>.patch.

Usage:
  python3 tools/gen_patches.py             # regenerate patches/
  python3 tools/gen_patches.py --verify    # SHA-roundtrip check
  python3 tools/gen_patches.py --check     # git apply --check
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import subprocess
import sys
import tempfile

# Reuse the patch parser from the migration tool so the body-extraction logic
# in --verify mode stays identical to what produced overlay/ in the first place.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_patch_to_overlay as importer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_BASELINE = os.path.join(ROOT, "vllm")
OVERLAY_ROOT = os.path.join(ROOT, "overlay", "vllm")
PATCHES_DIR = os.path.join(ROOT, "patches")
MANIFEST_PATH = os.path.join(PATCHES_DIR, "MANIFEST.txt")
SOURCE_PATH = os.path.join(PATCHES_DIR, "SOURCE.txt")

UPSTREAM_TAG = "v0.25.1"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def git_blob_sha(content_bytes: bytes) -> str:
    """Full git blob SHA-1 hex digest for the given byte content."""
    return hashlib.sha1(
        b"blob " + str(len(content_bytes)).encode("ascii") + b"\0" + content_bytes
    ).hexdigest()


def git_file_mode(path: str) -> str:
    """Return the git mode string for a file: '100755' if any executable bit
    is set, else '100644'. Symlinks (120000) are not supported by this
    workflow. Caller must ensure the path exists."""
    st = os.lstat(path)
    if os.path.islink(path):
        return "120000"
    if st.st_mode & 0o111:
        return "100755"
    return "100644"


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def read_baseline_bytes(repo_path: str) -> bytes:
    """Read the v0.25.1 baseline body for a repo-relative path.

    Returns b"" if the path does not exist in the baseline (new-file case).
    """
    full = os.path.join(VLLM_BASELINE, repo_path)
    if not os.path.exists(full):
        return b""
    return read_bytes(full)


def patch_name_for(repo_path: str) -> str:
    """R7: patches/vllm-<path-with-slashes-replaced-by-dashes>.patch."""
    dashed = repo_path.replace("/", "-")
    return f"vllm-{dashed}.patch"


def patch_to_repo_path(patch_name: str) -> str | None:
    """Inverse of patch_name_for. Returns None if the name is not a patch."""
    if not patch_name.startswith("vllm-") or not patch_name.endswith(".patch"):
        return None
    inner = patch_name[len("vllm-"):-len(".patch")]
    return inner.replace("-", "/")


# --------------------------------------------------------------------------- #
# Diff generation (hand-rolled, git-compatible)
# --------------------------------------------------------------------------- #

def _fmt_hunk_range(start_0idx: int, length: int) -> str:
    """Format a unified-diff hunk range. Git uses 1-indexed start for
    non-empty ranges; for empty ranges the convention is `X,0` where X is
    the line number preceding the insertion point (0 if at start of file)."""
    if length == 0:
        return f"{start_0idx},0"
    if length == 1:
        return f"{start_0idx + 1}"
    return f"{start_0idx + 1},{length}"


def _emit_line(out: list[str], prefix: str, line: str) -> None:
    """Emit a hunk body line with proper newline / no-newline handling."""
    if line.endswith("\n"):
        out.append(prefix + line)
    else:
        out.append(prefix + line + "\n")
        out.append("\\ No newline at end of file\n")


def make_diff(baseline_bytes: bytes, overlay_bytes: bytes,
              repo_path: str, is_new_file: bool, mode: str = "100644") -> bytes:
    """Return a git-style unified diff for repo_path.

    Layout matches what `git diff` emits: `diff --git`, optional
    `new file mode <mode>`, `index <old>..<new> <mode>`, `---`, `+++`,
    then one or more `@@ ... @@` hunks. The mode is the post-apply file
    mode (typically the baseline's mode for modified files, or the
    overlay's mode for new files).
    """
    baseline_text = baseline_bytes.decode("utf-8", errors="surrogateescape")
    overlay_text = overlay_bytes.decode("utf-8", errors="surrogateescape")
    baseline_lines = baseline_text.splitlines(keepends=True)
    overlay_lines = overlay_text.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(a=baseline_lines, b=overlay_lines,
                                      autojunk=False)
    body: list[str] = []
    for group in matcher.get_grouped_opcodes(3):
        first_tag, i1, _, j1, _ = group[0]
        _, _, i2, _, j2 = group[-1]
        old_len = i2 - i1
        new_len = j2 - j1
        body.append(f"@@ -{_fmt_hunk_range(i1, old_len)} "
                    f"+{_fmt_hunk_range(j1, new_len)} @@\n")
        for tag, bi1, bi2, oi1, oi2 in group:
            if tag == "equal":
                for line in baseline_lines[bi1:bi2]:
                    _emit_line(body, " ", line)
            else:
                if tag in ("replace", "delete"):
                    for line in baseline_lines[bi1:bi2]:
                        _emit_line(body, "-", line)
                if tag in ("replace", "insert"):
                    for line in overlay_lines[oi1:oi2]:
                        _emit_line(body, "+", line)

    if not body:
        # No textual difference. overlay/ should only contain files that
        # differ from baseline (R2); an identical file is a stray that
        # should be removed. Raise so the operator notices — the caller
        # (regenerate) handles this without aborting the whole run.
        raise IdenticalOverlayError(repo_path)

    header: list[str] = [f"diff --git a/{repo_path} b/{repo_path}\n"]
    if is_new_file:
        header.append(f"new file mode {mode}\n")
    old_sha = "0000000" if is_new_file else git_blob_sha(baseline_bytes)[:7]
    new_sha = git_blob_sha(overlay_bytes)[:7]
    header.append(f"index {old_sha}..{new_sha} {mode}\n")
    header.append("--- " + ("/dev/null" if is_new_file else f"a/{repo_path}") + "\n")
    header.append(f"+++ b/{repo_path}\n")

    return ("".join(header) + "".join(body)).encode(
        "utf-8", errors="surrogateescape")


class IdenticalOverlayError(ValueError):
    """Raised when an overlay file is byte-identical to its baseline.

    R2 forbids this (overlay/ contains only files that differ). The
    generator treats this as a soft error: print a warning, skip the
    file, and continue — do NOT abort mid-loop and leave patches/ in a
    half-written state.
    """


# --------------------------------------------------------------------------- #
# Default mode: regenerate patches/
# --------------------------------------------------------------------------- #

def write_atomic(path: str, content: bytes) -> None:
    """Write content to path via temp file + rename for atomicity."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp.", suffix=os.path.basename(path),
        dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def upstream_v0251_sha() -> str:
    """Return the v0.25.1 commit SHA from the vllm/ baseline clone."""
    res = subprocess.run(
        ["git", "-C", VLLM_BASELINE, "rev-parse", UPSTREAM_TAG],
        capture_output=True, check=True)
    return res.stdout.decode().strip()


def regenerate() -> int:
    if not os.path.isdir(OVERLAY_ROOT):
        sys.exit(f"overlay/ missing: {OVERLAY_ROOT}")
    if not os.path.isdir(VLLM_BASELINE):
        sys.exit(f"vllm/ baseline missing: {VLLM_BASELINE}")

    # 1. Walk overlay/, build a map of repo_path -> overlay abspath.
    overlay_files: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(OVERLAY_ROOT):
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, OVERLAY_ROOT)
            repo_path = rel.replace(os.sep, "/")
            overlay_files[repo_path] = abs_path

    # 2. Compute and write each patch.
    written: list[str] = []
    skipped_identical: list[str] = []
    for repo_path in sorted(overlay_files):
        overlay_bytes = read_bytes(overlay_files[repo_path])
        baseline_path = os.path.join(VLLM_BASELINE, repo_path)
        baseline_bytes = read_baseline_bytes(repo_path)
        is_new_file = baseline_bytes == b"" and not os.path.exists(baseline_path)
        # Mode bits: for modified files, the patch does not change modes,
        # so the result mode is the baseline's mode. For new files, it is
        # the overlay's mode.
        if is_new_file:
            mode = git_file_mode(overlay_files[repo_path])
        else:
            mode = git_file_mode(baseline_path)
        try:
            diff_bytes = make_diff(baseline_bytes, overlay_bytes,
                                   repo_path, is_new_file, mode=mode)
        except IdenticalOverlayError:
            skipped_identical.append(repo_path)
            print(f"warning: overlay/vllm/{repo_path} is byte-identical to "
                  "the v0.25.1 baseline (R2 violation); skipping — remove "
                  "this file from overlay/ to silence.")
            continue
        patch_name = patch_name_for(repo_path)
        write_atomic(os.path.join(PATCHES_DIR, patch_name), diff_bytes)
        written.append(patch_name)

    # 3. Remove patches whose overlay source is gone.
    expected = set(written)
    for existing in os.listdir(PATCHES_DIR):
        if not existing.endswith(".patch"):
            continue
        if existing == ".gitkeep":
            continue
        if existing not in expected:
            os.unlink(os.path.join(PATCHES_DIR, existing))
            print(f"removed orphan patch: {existing}")

    # 4. Rewrite MANIFEST and SOURCE atomically.
    manifest_body = "\n".join(sorted(expected)) + "\n"
    manifest_header = (
        f"# Files emitted by tools/gen_patches.py for the {UPSTREAM_TAG} "
        "lineage.\n"
        "# One patch per modified vLLM file under overlay/vllm/. Sorted; "
        "do not hand-edit\n"
        "# (regenerate via `python3 tools/gen_patches.py`).\n"
    )
    write_atomic(MANIFEST_PATH,
                 (manifest_header + manifest_body).encode("utf-8"))

    sha = upstream_v0251_sha()
    source_body = (
        f"# Upstream baseline for patches/ in this lineage.\n"
        f"# Tag: {UPSTREAM_TAG}\n"
        f"# Commit SHA: {sha}\n"
        f"# Read-only: do not modify the vllm/ checkout (see AGENTS.md).\n"
        f"{sha}\n"
    )
    write_atomic(SOURCE_PATH, source_body.encode("utf-8"))

    new_count = sum(1 for p in expected
                    if _patch_is_new_file(os.path.join(PATCHES_DIR, p)))
    mod_count = len(expected) - new_count
    print(f"wrote {len(expected)} patches ({new_count} new-file, "
          f"{mod_count} modified) into patches/")
    print(f"MANIFEST.txt + SOURCE.txt updated ({sha[:12]} @ {UPSTREAM_TAG})")
    return 0


def _patch_is_new_file(patch_path: str) -> bool:
    with open(patch_path, "rb") as f:
        head = f.read(512)
    return b"\nnew file mode 100644\n" in head


# --------------------------------------------------------------------------- #
# --verify mode: SHA-roundtrip
# --------------------------------------------------------------------------- #

def verify() -> int:
    if not os.path.isdir(PATCHES_DIR):
        sys.exit(f"patches/ missing: {PATCHES_DIR}")
    patches = sorted(p for p in os.listdir(PATCHES_DIR)
                     if p.endswith(".patch"))
    if not patches:
        sys.exit("patches/ is empty — run `python3 tools/gen_patches.py` first.")

    failures = 0
    for name in patches:
        patch_path = os.path.join(PATCHES_DIR, name)
        repo_path = patch_to_repo_path(name)
        if repo_path is None:
            print(f"[verify] {name}: cannot derive repo path from filename")
            failures += 1
            continue

        overlay_path = os.path.join(OVERLAY_ROOT, repo_path)
        if not os.path.exists(overlay_path):
            print(f"[verify] {name}: overlay source missing at {overlay_path}")
            failures += 1
            continue

        # Parse the patch, reconstruct the modified body, compute its blob SHA.
        patch_bytes = read_bytes(patch_path)
        try:
            blocks = importer.parse_patch(patch_bytes)
        except Exception as e:
            print(f"[verify] {name}: parse failed: {e}")
            failures += 1
            continue
        if len(blocks) != 1:
            print(f"[verify] {name}: expected 1 diff block, got {len(blocks)}")
            failures += 1
            continue
        block = blocks[0]
        if block.path != repo_path:
            print(f"[verify] {name}: block path {block.path!r} != "
                  f"derived {repo_path!r}")
            failures += 1
            continue
        try:
            if block.is_new_file:
                body_text = importer.reconstruct_new_file(block.hunks,
                                                          block.path)
            else:
                baseline_bytes = read_baseline_bytes(repo_path)
                baseline_text = baseline_bytes.decode(
                    "utf-8", errors="surrogateescape")
                body_text = importer.apply_hunks_to_baseline(
                    baseline_text, block.hunks, block.path)
        except Exception as e:
            print(f"[verify] {name}: reconstruct failed: {e}")
            failures += 1
            continue

        body_bytes = body_text.encode("utf-8", errors="surrogateescape")
        body_sha = git_blob_sha(body_bytes)
        overlay_bytes = read_bytes(overlay_path)
        overlay_sha = git_blob_sha(overlay_bytes)

        if body_sha != overlay_sha:
            print(f"[verify] {name}: SHA mismatch — patch reconstructs "
                  f"{body_sha[:12]} but overlay is {overlay_sha[:12]}")
            failures += 1
            continue

    if failures:
        print(f"\n[verify] FAILED: {failures}/{len(patches)} patches did not "
              "roundtrip-verify.")
        return 1
    print(f"[verify] OK: {len(patches)}/{len(patches)} patches roundtrip-"
          "verified against overlay/.")
    return 0


# --------------------------------------------------------------------------- #
# --check mode: git apply --check
# --------------------------------------------------------------------------- #

def check() -> int:
    if not os.path.isdir(PATCHES_DIR):
        sys.exit(f"patches/ missing: {PATCHES_DIR}")
    if not os.path.isdir(VLLM_BASELINE):
        sys.exit(f"vllm/ baseline missing: {VLLM_BASELINE}")
    patches = sorted(p for p in os.listdir(PATCHES_DIR)
                     if p.endswith(".patch"))
    if not patches:
        sys.exit("patches/ is empty — run `python3 tools/gen_patches.py` first.")

    failures = 0
    for name in patches:
        patch_path = os.path.join(PATCHES_DIR, name)
        res = subprocess.run(
            ["git", "-C", VLLM_BASELINE, "apply", "--check", patch_path],
            capture_output=True)
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="surrogateescape").strip()
            print(f"[check] {name}: does NOT apply cleanly\n  {err}")
            failures += 1
    if failures:
        print(f"\n[check] FAILED: {failures}/{len(patches)} patches do not "
              "apply against v0.25.1 baseline.")
        return 1
    print(f"[check] OK: {len(patches)}/{len(patches)} patches apply cleanly "
          "against v0.25.1 baseline.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="SHA-roundtrip: re-extract each patch's modified "
                           "body and assert its git blob SHA matches the "
                           "overlay file's SHA.")
    mode.add_argument("--check", action="store_true",
                      help="git apply --check each patch against the v0.25.1 "
                           "baseline in vllm/.")
    args = ap.parse_args()

    if args.verify:
        return verify()
    if args.check:
        return check()
    return regenerate()


if __name__ == "__main__":
    sys.exit(main())
