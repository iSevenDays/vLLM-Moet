# AGENTS.md — the commit contract

Multiple agents and humans work on this project concurrently, often in the
**same checkout**. Work has been lost here three separate times by editing
the distribution patch without verifying the result. Read this before your
first commit; every rule below traces back to a real incident.

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

## Editing vLLM code — the patch is the source of truth

The vLLM-Moet project does not ship a fork branch. The patch file
`patch/vllm-moet-v0.25.1.patch` IS the source of truth for every modification
the project makes to upstream vLLM v0.25.1. It is edited directly — by hand
or by agent — under the discipline below. There is no fork clone, no
regeneration step, no `--update` mode that produces the patch from anything
other than the previous patch plus an edit.

### The patch-direct edit procedure

1. Locate the file's hunk in `patch/vllm-moet-v0.25.1.patch`. New files
   use the `new file mode 100644 / index 0000000..<sha> / --- /dev/null /
   +++ b/<file> / @@ -0,0 +1,N @@` form. Modified files use the standard
   unified diff form — update hunks in place, preserving context lines.
2. Replace the body with the modified file content (every line `+`
   prefixed). Update the `@@ -0,0 +1,N @@` count and the `index` blob SHA
   (abbreviated git blob hash: `hashlib.sha1(b"blob " +
   str(len(content_bytes)).encode() + b"\0" + content_bytes).hexdigest()[:7]`).
3. **Roundtrip-verify**: re-extract the body from the rewritten patch and
   confirm its git blob SHA equals the modified file's git blob SHA.
   **No SHA match, no commit.**
4. Update `patch/FILES-v0.25.1.txt` if the file set changed.
5. Commit with a note in the body naming the file(s) touched.

### Why SHA-roundtrip is the load-bearing discipline

Three separate times a line of work was lost because the patch drifted from
what an agent or human claimed about it (Kimi-K2.7, DSpark, stream-build —
see "The incidents" below). The SHA-roundtrip check catches this class of
drift AT EDIT TIME, before the work lands. Skip it and the next reader of
the patch cannot tell which hunks are real.

A future plan
(`docs/plans/2026-07-20-001-refactor-overlay-patch-workflow-plan.md`)
replaces this manual procedure with a `tools/gen_patches.py` generator that
enforces SHA-roundtrip automatically. Until that plan ships, the manual
procedure above is the workflow.

### `tools/check_patch_files.py` — what it actually does

The tool has two modes:
- Default (no args): verifies `patch/FILES-v0.25.1.txt` matches the file
  list inside `patch/vllm-moet-v0.25.1.patch`. Runs anywhere; no fork clone
  needed. This is what CI bench-lint calls.
- `--update`: refuses to run. The mode was designed to regenerate the patch
  from a fork branch that does not exist. The refusal is explicit.

## Where a change goes

| change | commit where | must also update |
|---|---|---|
| vLLM runtime code (`moe_w2_*`, loaders, runner, attention, …) | `patch/vllm-moet-v0.25.1.patch` directly (patch-direct procedure above) | `patch/FILES-v0.25.1.txt` if the file set changed |
| SASS kernels / cubins | `kernels/` | a `kernels/MANIFEST.md` row (generator + validation status) |
| serve configs | `bench/recipes/` | `bench/models.yaml`, `bench/matrix.yaml`; run `bench/runner/lint.py` |
| bench results | `bench/results/<release>/` | `bench/runner/render.py` — the README table and per-release report are **generated**; never hand-edit the marked README block |
| docs | `docs/`, README outside the generated block | — |
| session notes / handoffs / experiment scraps | `internal/` (gitignored, stays local) | never into `docs/` |

Never commit: wheels (`patch/*.whl` is ignored), checkpoints, expert packs,
smoke results (`bench/results/smoke/`).

## Concurrency — several agents, one checkout

- `git status` before you start. Dirty paths you did not create belong to
  another live session: **leave them alone**. No `git add -A`, no
  `git commit -a`, no `git stash`, no `git checkout --` / `git reset` over
  someone else's files, ever.
- Stage **explicit paths only**: `git add <file> <file> …`.
- `main` is a shared trunk: no amending commits you did not just create,
  no rebase, no force-push, no history rewrite.
- The `patch/` trio (patch, `FILES-v0.25.1.txt`, `SOURCE-v0.25.1.txt`)
  changes ONLY via the patch-direct procedure above. Every edit
  roundtrip-verifies its own SHA before commit — see the procedure.
- A pre-commit hook in dev-box checkouts runs the patch guard whenever
  `patch/` is staged. Do not bypass it with `--no-verify`.
- Commit identity: no global git identity is configured on this box —
  pass the session identity per command, matching the existing history:

  ```bash
  git -c user.name=vllm-moet -c user.email=moet@local commit ...
  ```

  Upstream PRs use the user's public identity + DCO instead — see
  `internal/UPSTREAM_PRS.md`. Never edit git config.

## Pre-push checklist (mirrors CI `bench-lint`)

```bash
python3 tools/check_patch_files.py       # FILES-v0.25.1.txt <-> patch
python3 bench/runner/lint.py             # recipes/boxes/suites/results schemas
python3 bench/runner/render.py --check   # README table == committed results
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
  unchanged file set are invisible to `FILES-v0.25.1.txt` — that class is
  what the SHA-roundtrip discipline catches.

**A failing guard means work would be lost.** The fix is always to redo
the edit with roundtrip verification — never to edit `FILES-v0.25.1.txt`
or `SOURCE-v0.25.1.txt` into agreement with a drifted patch.

## Commit style

Match `git log`. Subject: declarative, what ships / what changed, no
prefixes. Body: the why, the evidence (measured numbers, test verdicts),
and the file(s) touched by the patch-direct edit.
