# AGENTS.md — the commit contract

Multiple agents and humans work on this project concurrently, often in the
**same checkout**. Work has been lost here three separate times by editing
the distribution patch without verifying the result. Read this before your
first commit; every rule below traces back to a real incident. The
SHA-roundtrip discipline that the three incidents motivated is now
tool-enforced by `tools/gen_patches.py --verify` — the rule still lives
in your head, but the gate runs in code.

## Serving rule — the quantization cache is specific to the GPU count and the residency

Read this rule before you serve on more than one GPU. Read this rule before you change the
residency.

The engine caches the 2-bit quantization for one configuration only. The cache depends on two
values:
- the tensor-parallel size N (the number of GPUs);
- the residency (`host` or `gpu`).

`host` residency writes a pack file `base.rank<i>of<N>.pack`. `gpu` residency
(`VLLM_MOE_W2_BASE_CACHE_GB=0`) writes the plane cache. A cache from one configuration does
not apply to a different configuration. A cache from TP1 does not apply to a TP2 run. A cache
from `host` does not apply to a `gpu` run. In these conditions the engine does a new
quantization. A new quantization needs 15 to 20 minutes.

Obey these steps:
1. Select the GPU count and the residency before the first boot.
2. To serve on N GPUs, do the first-run quantization at `TP=N`.
3. If the host has much VRAM and little RAM (for example, 2 x 48 GB VRAM and 20 to 30 GB RAM),
   use `gpu` residency (`RESIDENCY=gpu`). The base then stays on the GPUs. This mode also
   prevents the host-RAM out-of-memory condition.
4. Keep `VLLM_ENGINE_READY_TIMEOUT_S` more than the quantization time. The default value is
   600 seconds. This value is too small and stops the quantization. The launcher sets 1800
   seconds.

For the launch commands and more data, refer to `docker/serve_sm89_ds4.sh`, header section 2,
and the TECHNICAL NOTES.

## Editing vLLM code — overlay/ is the source of truth

The vLLM-Moet project ships its modifications to upstream vLLM as a set of
per-file unified-diff patches under `patches/`. The **source of truth** is
the `overlay/vllm/<repo-relative-path>` tree — full files, edited directly.
`patches/` is generated from `overlay/` diffed against the read-only
`vllm/` baseline pinned at the `v0.25.1` tag (commit `752a3a504`). Never
hand-edit `patches/`; always edit `overlay/` and regenerate.

There is no fork branch. There is no regen-from-fork step. The hallucinated
fork-clone workflow that older versions of this file described does not
exist.

### The workflow

1. **Edit `overlay/vllm/<path>`.** If the file is new (does not exist in
   the v0.25.1 baseline), just create it under `overlay/vllm/<path>`. If
   it is a modification of a baseline file, copy
   `vllm/<path>` → `overlay/vllm/<path>` first, then edit the overlay
   copy.
2. **Regenerate patches.** `python3 tools/gen_patches.py` walks `overlay/`,
   diffs each file against the baseline, writes one patch per file to
   `patches/vllm-<dashed-path>.patch`, removes any patch whose overlay
   source is gone, and rewrites `patches/MANIFEST.txt` and
   `patches/SOURCE.txt` atomically.
3. **Verify roundtrip.** `python3 tools/gen_patches.py --verify`
   re-extracts each patch's modified body and asserts its git blob SHA
   equals the overlay file's git blob SHA. **SHA mismatch, no commit.**
   This is the load-bearing discipline — see "The incidents" below.
4. **Verify applicability.** `python3 tools/gen_patches.py --check` runs
   `git apply --check` per patch against the v0.25.1 baseline.
5. **Commit `overlay/vllm/<path>` + `patches/<name>.patch` +
   `patches/MANIFEST.txt` together.** If the file set is unchanged but a
   file body changed, only the overlay file + its single patch are touched
   (MANIFEST may not change at all — commit it together anyway if the
   generator rewrote it; the byte-stable case is also fine).

### Why SHA-roundtrip is the load-bearing discipline

Three separate times a line of work was lost because the artifact drifted
from what an agent or human claimed about it (Kimi-K2.7, DSpark,
stream-build — see "The incidents" below). The SHA-roundtrip check catches
this class of drift AT EDIT TIME, before the work lands. Skip it and the
next reader of the patch set cannot tell which hunks are real.

`tools/gen_patches.py --verify` makes the SHA-roundtrip check
tool-enforced: it re-parses each patch, reconstructs the modified file
body, computes its git blob SHA, and compares against the overlay file's
git blob SHA. The old patch-direct fallback procedure (hand-computed
`hashlib.sha1(b"blob " + len + b"\0" + content)`) is now native to the
generator and runs on every regeneration.

### `tools/gen_patches.py` — what it does

Three modes:
- **Default (no flag):** regenerates `patches/*.patch`, `patches/MANIFEST.txt`,
  and `patches/SOURCE.txt` from `overlay/` + the v0.25.1 baseline.
- `--verify`: SHA-roundtrip check on every patch (the load-bearing gate).
- `--check`: `git apply --check` per patch against the v0.25.1 baseline.

### `tools/check_patch_files.py` — thin wrapper

`tools/check_patch_files.py` is now a thin wrapper that invokes
`tools/gen_patches.py --verify` and additionally checks `patches/MANIFEST.txt`
matches the actual `patches/*.patch` set. Existing CI hooks and pre-push
checklists that call it continue to work; the broken `--update` mode is
gone (the workflow it described never existed).

## Where a change goes

| change | commit where | must also update |
|---|---|---|
| vLLM runtime code (`moe_w2_*`, loaders, runner, attention, …) | `overlay/vllm/<path>` directly; regenerate via `tools/gen_patches.py` | `patches/<name>.patch` + `patches/MANIFEST.txt` (regenerated); commit together |
| SASS kernels / cubins | `kernels/` | a `kernels/MANIFEST.md` row (generator + validation status) |
| serve configs | `bench/recipes/` | `bench/models.yaml`, `bench/matrix.yaml`; run `bench/runner/lint.py` |
| bench results | `bench/results/<release>/` | `bench/runner/render.py` — the README table and per-release report are **generated**; never hand-edit the marked README block |
| docs | `docs/`, README outside the generated block | — |
| session notes / handoffs / experiment scraps | `internal/` (gitignored, stays local) | never into `docs/` |

Never commit: wheels (any `*.whl` is ignored), checkpoints, expert packs,
smoke results (`bench/results/smoke/`).

## Concurrency — several agents, one checkout

- `git status` before you start. Dirty paths you did not create belong to
  another live session: **leave them alone**. No `git add -A`, no
  `git commit -a`, no `git stash`, no `git checkout --` / `git reset` over
  someone else's files, ever.
- Stage **explicit paths only**: `git add <file> <file> …`.
- `main` is a shared trunk: no amending commits you did not just create,
  no rebase, no force-push, no history rewrite.
- `overlay/` + `patches/` change ONLY via the workflow above. Every edit
  roundtrip-verifies its own SHA before commit — see the procedure.
- A pre-commit hook in dev-box checkouts runs the patch guard whenever
  `overlay/` or `patches/` is staged. Do not bypass it with `--no-verify`.
- The `vllm/` baseline clone is **read-only**. The workflow assumes
  `vllm/` is at the `v0.25.1` tag (commit `752a3a504`). Never modify
  files inside `vllm/`; never check out a different ref. If a future
  lineage (e.g. v0.26) needs to land, it gets its own `overlay-v0.26/`
  and `patches-v0.26/` trees.
- Commit identity: plain `git commit`, attributed to whatever
  `git config user.name` / `user.email` returns on your box.

  Upstream PRs use the same identity + DCO — see `internal/UPSTREAM_PRS.md`.
  Never edit git config.

## Pre-push checklist (mirrors CI `bench-lint`)

```bash
python3 tools/gen_patches.py --verify     # overlay SHA == patch SHA
python3 tools/gen_patches.py --check      # git apply --check on v0.25.1
python3 tools/check_patch_files.py        # MANIFEST == patches/*.patch set
python3 bench/runner/lint.py              # recipes/boxes/suites/results schemas
python3 bench/runner/render.py --check    # README table == committed results
```

Plus `python3 docker/serve_recipe.py <recipe> --print` if you touched
recipes or the launcher.

## The incidents these rules come from

- **Kimi-K2.7 bring-up** — landed in the patch without traceable roundtrip
  verification; an overwrite step erased it; caught by hand, folded back
  in `81c1b34`.
- **DSpark backport** — `ad7f29a` introduced a patch body that lacked the
  DSpark line and silently dropped it; repaired by restoring the union of
  both edits in `8aff1b7`.
- **Stream-build hunks** — lost to a merge-side resolution inside the patch
  file; restored by hand in `8f50e57`. Hunk-level losses inside an
  unchanged file set are invisible to a manifest check — that class is
  what the SHA-roundtrip discipline catches.

**A failing guard means work would be lost.** The fix is always to redo
the edit with roundtrip verification — never to commit a hand-edited
patch into agreement with a drifted overlay, and never to edit
`patches/MANIFEST.txt` or `patches/SOURCE.txt` to paper over a mismatch.

## Commit style

Match `git log`. Subject: declarative, what ships / what changed, no
prefixes. Body: the why, the evidence (measured numbers, test verdicts),
and the file(s) touched by the overlay edit + the regenerated patch.
