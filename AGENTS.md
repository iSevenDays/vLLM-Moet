# AGENTS.md — the commit contract

Multiple agents and humans work on this project concurrently, often in the
**same checkout**. Work has been lost here three separate times by treating
the generated patch as a source file. Read this before your first commit;
every rule below traces back to a real incident.

## The iron rule

Every `patch/vllm-moet-<tag>.patch` is a **generated artifact**:
byte-for-byte `git diff <tag> moet-<tag>` from the vllm fork clone. One
lineage ships — `v0.25.1` (branch `moet-v0.25.1`, fingerprints
`FILES-v0.25.1.txt`/`SOURCE-v0.25.1.txt`); the retired `v0.24.0`
lineage was removed 2026-07-19 (only the v0.25.1 image is deployed;
its history stays in git). A patch is
**never edited by hand**, never patched incrementally, never regenerated
from anything but its fork branch. The only sanctioned way to change one:

```bash
python3 tools/check_patch_files.py --update [tag ...]   # default: all
```

If your change touches vLLM code, it goes to the **fork branch first**; the
patch is derived afterwards. A change that exists only inside the patch file
WILL be erased by somebody's next regeneration.

### When the fork clone is not reachable (the patch-direct fallback)

The fork clone lives on the **dev box** only (path `/workspace/vllm-v0.24.0`,
or wherever `VLLM_MOET_FORK` points). Other environments — CI, contributor
Macs, and any deploy/build host that just pulls this repo to drive a
Docker build — do **not** have it. A `vllm/` subdirectory, if present at
all on a non-dev box, is a locally-added IDE reference (gitignored; not
part of this repo). It is never the fork, and you cannot regen from it.

**The patch is the actual distribution mechanism.** The Dockerfile
applies `patch/vllm-moet-<tag>.patch` to the upstream image at build
time. When you are on a box without the fork clone and a vLLM code
change is needed, editing the patch directly is the **documented
fallback**, not a violation. The discipline below keeps it safe —
matching the regen-from-fork output byte-for-byte so the next dev-box
regen does not erase it.

Procedure:

1. Edit the source in a staging tree (e.g. `staging/<topic>/…`) by
   extracting the current file body out of the patch, modifying it, and
   validating syntax (`python3 -c "import ast; ast.parse(open(…).read())"`).
   Keep the pristine extraction as `<file>.orig` for diff review.
2. Locate the file's hunk in `patch/vllm-moet-<tag>.patch`. New files
   use the `new file mode 100644 / index 0000000..<sha> / --- /dev/null /
   +++ b/<file> / @@ -0,0 +1,N @@` form. Modified files use the standard
   unified diff form — update hunks in place, preserving context lines.
3. Replace the body with the modified file content (every line `+`
   prefixed). Update the `@@ -0,0 +1,N @@` count and the `index` blob
   SHA (abbreviated git blob hash: `hashlib.sha1(b"blob " +
   str(len(content_bytes)).encode() + b"\0" + content_bytes).hexdigest()[:7]`).
4. **Roundtrip-verify**: re-extract the body from the rewritten patch
   and confirm its git blob SHA equals the modified file's git blob SHA.
   No SHA match, no commit.
5. Note in the commit body that this is a patch-direct edit pending
   fork-branch sync, and name the file(s) touched. The next dev-box
   session must `git apply -3` the commit's patch hunk onto
   `moet-v0.25.1` so a regen produces a byte-equal patch.

The iron rule still applies wherever the fork clone IS reachable:
prefer regen over hand-edit there. The fallback exists so work does not
stall on boxes that cannot reach the fork.

## The two repos

| repo | path | branch | role |
|---|---|---|---|
| **vLLM-Moet** (this one, public) | wherever the checkout lives on the current box | `main` | publication: generated patch, kernels + cubins, bench system, docs |
| **vllm fork clone** (dev box only) | `/workspace/vllm-v0.24.0` (path is historical) | `moet-v0.25.1` | source of truth for ALL vLLM code *where reachable*; remotes: `fork` = `kacper-daftcode/vllm`, `origin` = `vllm-project/vllm`. Not present on CI, contributor Macs, or deploy/build hosts — see the patch-direct fallback above. |

Upstream-PR branches and experiments live in worktrees off the fork clone
(`git -C /workspace/vllm-v0.24.0 worktree list`).

`moet-v0.25.1` is the ship lineage: everything committed there is meant to
ship in the next patch regen. Park half-done work on a side branch or
worktree, not on `moet-v0.25.1`.

## Where a change goes

| change | commit where | must also update |
|---|---|---|
| vLLM runtime code (`moe_w2_*`, loaders, runner, attention, …) | fork branch `moet-v0.25.1` | regen the patch here — procedure below |
| SASS kernels / cubins | `kernels/` | a `kernels/MANIFEST.md` row (generator + validation status) |
| serve configs | `bench/recipes/` | `bench/models.yaml`, `bench/matrix.yaml`; run `bench/runner/lint.py` |
| bench results | `bench/results/<release>/` | `bench/runner/render.py` — the README table and per-release report are **generated**; never hand-edit the marked README block |
| docs | `docs/`, README outside the generated block | — |
| session notes / handoffs / experiment scraps | `internal/` (gitignored, stays local) | never into `docs/` |

Never commit: wheels (`patch/*.whl` is ignored), checkpoints, expert packs,
smoke results (`bench/results/smoke/`).

## Shipping a vLLM code change — the procedure

1. **Commit on the fork branch** (`/workspace/vllm-v0.24.0`,
   `moet-v0.25.1`). If the remote may have moved, fetch and merge first —
   the regen tool refuses to run when the local branch is missing pushed
   commits.
2. **Regenerate** from this repo:

   ```bash
   python3 tools/check_patch_files.py --update
   ```

   This rewrites the patch from the branch tip and updates the two
   committed fingerprints: `patch/FILES.txt` (file list) and
   `patch/SOURCE.txt` (the fork SHA the patch was generated from). It
   refuses to move `SOURCE.txt` backwards along the branch, so a
   regeneration can never roll back work that already shipped.
3. **Review `git diff patch/`.** An entry *vanishing* from `FILES.txt`
   means the patch carried work that never reached the fork branch —
   someone skipped step 1. **Stop and merge that work into the branch**;
   never ship the loss, never `--update` a second time to silence it.
4. **Validate what the change class requires.** Byte-exact generation from
   the branch already guarantees `git apply --check` on the tag. For
   GPU-relevant changes run the relevant suites
   (`tools/test_moe_w2_forward.py`, `tools/test_store_backends.py`, the
   three-tier tests) and put the results in the commit message, as the
   existing history does.
5. **Commit both repos, cross-referenced.** The vLLM-Moet commit that ships
   a regen names the fork SHA, following the established style:

   > `Three-tier starvation fix ships: step-scoped seen windows (vllm 9736e4d34)`

6. **Push both together** (`fork moet-v0.25.1` + `origin main`) once the
   pre-push checklist passes — or leave both unpushed. Avoid a lasting
   state where only one side is pushed.

## Concurrency — several agents, one checkout

- `git status` before you start. Dirty paths you did not create belong to
  another live session: **leave them alone**. No `git add -A`, no
  `git commit -a`, no `git stash`, no `git checkout --` / `git reset` over
  someone else's files, ever.
- Stage **explicit paths only**: `git add <file> <file> …`.
- `main` and `moet-v0.25.1` are shared trunks: no amending commits you did
  not just create, no rebase, no force-push, no history rewrite.
- The `patch/` trio (patch, `FILES-v0.25.1.txt`, `SOURCE-v0.25.1.txt`)
  changes **only** via `--update` on boxes where the fork clone is
  reachable. On boxes without the fork clone, the patch-direct fallback
  above applies — but `FILES`/`SOURCE` fingerprints must still be
  reconciled (see the fallback procedure; do not bump `SOURCE.txt`
  without a fork SHA to name).
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
python3 tools/check_patch_files.py       # patch <-> FILES.txt <-> SOURCE.txt
python3 bench/runner/lint.py             # recipes/boxes/suites/results schemas
python3 bench/runner/render.py --check   # README table == committed results
```

Plus `python3 docker/serve_recipe.py <recipe> --print` if you touched
recipes or the launcher.

## The incidents these rules come from

- **Kimi-K2.7 bring-up** — landed in the patch without its commits reaching
  the fork branch; the next regeneration erased it; caught by hand, folded
  back in `81c1b34`.
- **DSpark backport** — `ad7f29a` regenerated from a branch that lacked the
  DSpark line and silently dropped it; repaired by merging DSpark *into the
  generating branch* and regenerating the union (`8aff1b7`).
- **Stream-build hunks** — lost to a merge-side resolution inside the patch
  file; restored by hand in `8f50e57`. Hunk-level losses inside an
  unchanged file set are invisible to `FILES.txt` — that class is what the
  `SOURCE.txt` byte-verification catches.

**A failing guard means work would be lost.** The fix is always to move the
work onto the fork branch — never to edit the patch, `FILES.txt` or
`SOURCE.txt` into agreement.

## Commit style

Match `git log`. Subject: declarative, what ships / what changed, no
prefixes. Body: the why, the evidence (measured numbers, test verdicts),
and for regens the fork SHA. On the fork branch, upstream-style component
prefixes are fine (`DCP DSA indexer: …`).
