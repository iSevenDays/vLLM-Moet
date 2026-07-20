#!/usr/bin/env python3
"""One-time migration: split the legacy single-file patch into overlay/ files.

Historical tool. Ran once at the cutover from `patch/vllm-moet-v0.25.1.patch`
to the `overlay/` + `patches/` workflow. Kept in-tree as the audit trail for
how `overlay/` got populated; future re-runs require `--force` and are not
part of the normal workflow (see AGENTS.md).

Input: the existing single-file patch (default `patch/vllm-moet-v0.25.1.patch`).
For each `diff --git a/<path> b/<path>` block:
  - new-file blocks (`new file mode 100644` + `--- /dev/null`): the overlay
    body is the concatenation of the hunk's `+`-prefixed lines.
  - modified-file blocks: read the baseline body from `vllm/<path>`, apply
    each hunk in order, write the result to `overlay/vllm/<path>`.

The output tree is the source of truth that `tools/gen_patches.py` later
diffs against the pinned `vllm/` baseline to regenerate the per-file patches.

Usage:
  python3 tools/import_patch_to_overlay.py [--force] [--dry-run] [--patch PATH]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATCH = os.path.join(ROOT, "patch", "vllm-moet-v0.25.1.patch")
VLLM_BASELINE = os.path.join(ROOT, "vllm")
OVERLAY_ROOT = os.path.join(ROOT, "overlay", "vllm")

DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    baseline_start: int  # 1-indexed line in baseline where this hunk begins
    baseline_count: int  # number of baseline lines the hunk consumes
    body: list[str] = field(default_factory=list)  # raw hunk body lines incl. prefix


@dataclass
class DiffBlock:
    path: str  # repo-relative path (the b/ side)
    is_new_file: bool
    hunks: list[Hunk] = field(default_factory=list)
    # The head of each `body` line in a hunk is one of: ' ', '+', '-', '\'.
    # '\' is the "\\ No newline at end of file" marker — handled at apply time.


def parse_patch(patch_bytes: bytes) -> list[DiffBlock]:
    """Parse a unified-diff patch into per-file DiffBlocks.

    We split on `diff --git` boundaries. Within each block we look for the
    optional `new file mode` marker and the first `@@ ... @@` hunk header.
    Trailing metadata (e.g. `-- ` signature, or the next `diff --git`) ends
    the block.
    """
    text = patch_bytes.decode("utf-8", errors="surrogateescape")
    lines = text.splitlines(keepends=False)
    blocks: list[DiffBlock] = []
    current: DiffBlock | None = None
    current_hunk: Hunk | None = None
    seen_hunks_for_current = False

    for line in lines:
        m = DIFF_HEADER_RE.match(line)
        if m:
            # Flush previous hunk
            if current and current_hunk is not None:
                current.hunks.append(current_hunk)
                current_hunk = None
            # Flush previous block
            if current is not None:
                blocks.append(current)
            path = m.group(2)
            # Some paths may be quoted with `"`. Rarely happens here; reject loudly.
            if path.startswith('"'):
                raise ValueError(f"Quoted paths unsupported in block header: {line!r}")
            current = DiffBlock(path=path, is_new_file=False)
            seen_hunks_for_current = False
            current_hunk = None
            continue

        if current is None:
            # Leading garbage before first diff --git; ignore.
            continue

        if line.startswith("new file mode"):
            current.is_new_file = True
            continue

        if line.startswith("deleted file mode"):
            raise ValueError(
                f"{current.path}: deleted-file mode is not supported by this "
                "importer — the v0.25.1 patch has none; refusing to guess."
            )

        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("index "):
            # Metadata lines we don't need at apply time (we read baseline
            # from vllm/<path> directly).
            continue

        if line.startswith("rename from") or line.startswith("rename to"):
            raise ValueError(
                f"{current.path}: rename hunks are not supported by this "
                "importer — the v0.25.1 patch has none; refusing to guess."
            )

        hm = HUNK_HEADER_RE.match(line)
        if hm:
            if current_hunk is not None:
                current.hunks.append(current_hunk)
            start = int(hm.group(1))
            count = int(hm.group(2)) if hm.group(2) is not None else 1
            current_hunk = Hunk(baseline_start=start, baseline_count=count)
            seen_hunks_for_current = True
            continue

        if seen_hunks_for_current and current_hunk is not None:
            # Hunk body line. Prefix must be ' ', '+', '-', or '\'.
            if not line:
                # Empty line in a hunk body — git emits these as a single
                # space-prefixed context line that some editors strip. Treat
                # an empty body line as a context line containing nothing.
                current_hunk.body.append(" ")
                continue
            prefix = line[0]
            if prefix in (" ", "+", "-", "\\"):
                current_hunk.body.append(line)
                continue
            # Anything else inside a hunk region ends the hunk (signature,
            # next-block preamble that did not start with `diff --git`, etc.)
            current.hunks.append(current_hunk)
            current_hunk = None
            seen_hunks_for_current = False
            # Fall through: treat as block-level noise (e.g. `-- version`).
            continue

        # Block-level preamble not inside a hunk; ignore.
        continue

    # Flush trailing state.
    if current is not None:
        if current_hunk is not None:
            current.hunks.append(current_hunk)
        blocks.append(current)

    return blocks


def apply_hunks_to_baseline(
    baseline_text: str, hunks: list[Hunk], path: str
) -> str:
    """Apply hunks to baseline_text and return the resulting text.

    Baseline lines are 1-indexed. The baseline is split keeping line endings
    so we can re-join without behavior changes. We operate on the bytes-like
    list of (line, ends_with_newline) by splitting with keepends.
    """
    baseline_lines = baseline_text.splitlines(keepends=True)
    result: list[str] = []
    baseline_idx = 0  # number of baseline lines consumed so far (0-indexed count)

    def consume_until(target_line_no: int) -> None:
        """Emit baseline lines (unchanged) until 1-indexed line == target."""
        nonlocal baseline_idx
        # target_line_no is 1-indexed; baseline_idx is count consumed.
        # We want to advance until baseline_idx == target_line_no - 1.
        while baseline_idx < target_line_no - 1 and baseline_idx < len(baseline_lines):
            result.append(baseline_lines[baseline_idx])
            baseline_idx += 1

    for hunk_no, hunk in enumerate(hunks, start=1):
        consume_until(hunk.baseline_start)
        # Now baseline_idx should == hunk.baseline_start - 1.
        # Apply hunk body lines.
        for body in hunk.body:
            if body == "":
                continue  # defensive; parser already coerces empties
            prefix = body[0]
            content = body[1:]
            # Re-insert the line terminator that splitlines() stripped from
            # the patch body. We always assume `\n` terminators inside the
            # patch (the patch's own line endings define this); if the
            # baseline uses a different terminator, the AE5 parity check in
            # U4 will catch it.
            line_with_nl = content + "\n"
            if prefix == " ":
                # Context line. Must match the next baseline line.
                if baseline_idx >= len(baseline_lines):
                    raise ValueError(
                        f"{path}: hunk {hunk_no} context line past EOF — "
                        "patch does not apply to this baseline."
                    )
                actual = baseline_lines[baseline_idx]
                if actual != line_with_nl and actual.rstrip("\n") != content:
                    raise ValueError(
                        f"{path}: hunk {hunk_no} context mismatch at "
                        f"baseline line {baseline_idx + 1}: "
                        f"expected {content!r}, baseline has "
                        f"{actual.rstrip(chr(10))!r}."
                    )
                result.append(actual)
                baseline_idx += 1
            elif prefix == "-":
                if baseline_idx >= len(baseline_lines):
                    raise ValueError(
                        f"{path}: hunk {hunk_no} removed line past EOF."
                    )
                actual = baseline_lines[baseline_idx]
                if actual != line_with_nl and actual.rstrip("\n") != content:
                    raise ValueError(
                        f"{path}: hunk {hunk_no} removed-line mismatch at "
                        f"baseline line {baseline_idx + 1}: "
                        f"expected {content!r}, baseline has "
                        f"{actual.rstrip(chr(10))!r}."
                    )
                baseline_idx += 1  # drop the line
            elif prefix == "+":
                result.append(line_with_nl)
            elif prefix == "\\":
                # `\ No newline at end of file` marker. Adjusts the trailing
                # newline state of the most recent emitted line. For
                # simplicity, when this marker appears, we strip the trailing
                # newline from the last emitted result line.
                # Heuristic: git emits `\ No newline...` immediately after
                # the line it refers to. We strip the trailing newline on
                # the last appended result line.
                if result:
                    result[-1] = result[-1].rstrip("\n")
                # If the marker refers to a baseline line we just consumed
                # (a `-` line followed by `\`), and the baseline actually
                # ends without newline, we need to also strip the newline
                # from any future context line we re-emit. In practice the
                # v0.25.1 patch does not exercise this corner; the AE5
                # parity check will catch any miss.
                continue
            else:
                raise ValueError(f"{path}: unexpected hunk prefix {prefix!r}")

    # Drain remaining baseline lines.
    while baseline_idx < len(baseline_lines):
        result.append(baseline_lines[baseline_idx])
        baseline_idx += 1

    return "".join(result)


def reconstruct_new_file(hunks: list[Hunk], path: str) -> str:
    """For new-file blocks, the body is just the `+` lines concatenated."""
    parts: list[str] = []
    saw_no_newline_marker = False
    for hunk in hunks:
        for body in hunk.body:
            if not body:
                continue
            prefix = body[0]
            content = body[1:]
            if prefix == "+":
                parts.append(content + "\n")
            elif prefix == " ":
                # New-file hunks should have no context lines, but be lenient.
                parts.append(content + "\n")
            elif prefix == "-":
                raise ValueError(
                    f"{path}: new-file hunk has a `-` line, which is "
                    "inconsistent with `new file mode`."
                )
            elif prefix == "\\":
                saw_no_newline_marker = True
    text = "".join(parts)
    if saw_no_newline_marker:
        text = text.rstrip("\n")
    return text


def read_baseline(path: str) -> str:
    full = os.path.join(VLLM_BASELINE, path)
    if not os.path.exists(full):
        # The patch references a path not in v0.25.1 baseline — treat as new
        # file (the caller should already route via is_new_file, but be safe).
        return ""
    with open(full, "r", encoding="utf-8", errors="surrogateescape") as f:
        return f.read()


def write_overlay(path: str, content: str, force: bool, dry_run: bool) -> bool:
    full = os.path.join(OVERLAY_ROOT, path)
    if os.path.exists(full) and not force:
        raise FileExistsError(
            f"{full} already exists — pass --force to overwrite, or remove "
            "overlay/ first."
        )
    if dry_run:
        return True
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(content)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patch", default=DEFAULT_PATCH,
                    help=f"path to single-file patch (default: {DEFAULT_PATCH})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing overlay/ files")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files that would be written; write nothing")
    args = ap.parse_args()

    if not os.path.exists(args.patch):
        sys.exit(f"patch not found: {args.patch}")
    if not os.path.isdir(VLLM_BASELINE):
        sys.exit(f"vllm/ baseline missing: {VLLM_BASELINE}")

    with open(args.patch, "rb") as f:
        patch_bytes = f.read()

    blocks = parse_patch(patch_bytes)
    if not blocks:
        sys.exit("no `diff --git` blocks parsed from patch; refusing.")

    new_count, mod_count = 0, 0
    written = 0
    for block in blocks:
        if block.is_new_file:
            new_count += 1
            content = reconstruct_new_file(block.hunks, block.path)
        else:
            mod_count += 1
            baseline = read_baseline(block.path)
            if not baseline and not block.hunks:
                # Empty baseline, no hunks — nothing to do. Skip.
                continue
            content = apply_hunks_to_baseline(baseline, block.hunks, block.path)
        if args.dry_run:
            print(f"would write overlay/vllm/{block.path} "
                  f"({len(content)} bytes, "
                  f"{'new' if block.is_new_file else 'modified'})")
        else:
            write_overlay(block.path, content, args.force, args.dry_run)
            print(f"overlay/vllm/{block.path} "
                  f"({len(content)} bytes, "
                  f"{'new' if block.is_new_file else 'modified'})")
        written += 1

    kind = "would write" if args.dry_run else "wrote"
    print(f"\n{kind} {written} files "
          f"({new_count} new, {mod_count} modified) into overlay/vllm/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
