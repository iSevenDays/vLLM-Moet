#!/usr/bin/env python3
"""Differential test: our compressor reference vs the CHECKPOINT's own math.

Why a THIRD implementation. `vllm/tests/kernels/test_compressor_kv_cache.py`
validates the fused compress kernel against `_reference_kv_compress_norm_rope`
in the same file -- but both were written by this port, so a shared misreading
of the architecture passes silently. The only independent source of truth is the
implementation shipped INSIDE the checkpoint:
`/root/models/DeepSeek-V4-Flash-0731/inference/model.py::Compressor`.

That reference is whole-sequence/batch shaped; ours is per-output-token and
paged. This test transcribes the checkpoint math directly (batch form, from
`Compressor.forward`'s start_pos==0 branch plus `overlap_transform`) and checks
the two agree on the structural semantics that actually matter:

  1. WHICH state entries each compressed output pools over (the ratio-4
     overlapping window: 4 entries from the PREVIOUS block using the first half
     of the 2*head_dim projection, then 4 from the CURRENT block using the
     second half).
  2. The softmax axis and the -inf/0 padding of block 0's absent overlap half.
  3. APE indexing (by the token's own within-block position).
  4. The RoPE POSITION index chosen per compressed output.

RoPE *convention* is deliberately shared between the two paths here (both use
the in-repo rotation helper) so a convention mismatch cannot masquerade as a
compression bug; the position index is asserted separately as an integer.

CPU-only, no GPU, no server, seconds per run:

  docker run --rm --entrypoint python3 \
    -v $PWD/tools/test_compressor_vs_checkpoint_ref.py:/opt/t.py:ro \
    -v $PWD/vllm/tests/kernels:/kernels:ro \
    vllm-moet-sm89:v0251 /opt/t.py
"""
import importlib.util
import sys
import types

import torch

HEAD_DIM = 512
ROPE_DIM = 64
EPS = 1e-6


def load_repo_reference(path="/kernels/test_compressor_kv_cache.py"):
    """Import `_reference_kv_compress_norm_rope` from the repo test module.

    The module does `from .test_fused_indexer_q_rope_quant import ...`, so it
    needs a package context; the sibling is stubbed because only the mxfp4
    path uses it and we call with return_full_cache=True.
    """
    pkg = types.ModuleType("kpkg")
    pkg.__path__ = ["/kernels"]
    sys.modules["kpkg"] = pkg
    sib = types.ModuleType("kpkg.test_fused_indexer_q_rope_quant")
    sib.quantize_to_mxfp4 = lambda *a, **k: None
    sys.modules["kpkg.test_fused_indexer_q_rope_quant"] = sib
    spec = importlib.util.spec_from_file_location(
        "kpkg.test_compressor_kv_cache", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._reference_kv_compress_norm_rope


# --------------------------------------------------------------------------
# Transcription of the CHECKPOINT's Compressor (inference/model.py), batch form
# --------------------------------------------------------------------------
def ckpt_overlap_transform(tensor, ratio, d, value):
    """Verbatim from the checkpoint's Compressor.overlap_transform.

    tensor [b, s, ratio, 2d] -> [b, s, 2*ratio, d]
      new[:, :, ratio:] = tensor[:, :, :, d:]      # current block, 2nd half
      new[:, 1:, :ratio] = tensor[:, :-1, :, :d]   # previous block, 1st half
    Block 0's overlap half keeps `value` (0 for kv, -inf for score).
    """
    b, s = tensor.shape[0], tensor.shape[1]
    new = tensor.new_full((b, s, 2 * ratio, d), value)
    new[:, :, ratio:] = tensor[:, :, :, d:]
    new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new


def ckpt_compress(kv, score, ape, rms_weight, ratio, cast_bf16_before_norm):
    """Checkpoint prefill branch (start_pos == 0), remainder == 0, bsz == 1.

        kv    = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + ape
        if overlap: kv, score = overlap_transform(...)
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = norm(kv.to(dtype))

    Returns (compressed_normed [nblk, d], rope_positions [nblk]).
    """
    overlap = ratio == 4
    d = HEAD_DIM
    kv = kv.unsqueeze(0).unflatten(1, (-1, ratio))          # [1,nblk,ratio,2d]
    score = score.unsqueeze(0).unflatten(1, (-1, ratio)) + ape
    if overlap:
        kv = ckpt_overlap_transform(kv, ratio, d, 0.0)
        score = ckpt_overlap_transform(score, ratio, d, float("-inf"))
    else:
        kv = kv[..., :d]
        score = score[..., :d]
    pooled = (kv * score.softmax(dim=2)).sum(dim=2).squeeze(0)   # [nblk, d]

    if cast_bf16_before_norm:
        # The checkpoint does `self.norm(kv.to(dtype))` -- it rounds to the
        # activation dtype BEFORE RMSNorm. Kept switchable so this ordering
        # detail cannot be mistaken for a structural disagreement.
        pooled = pooled.to(torch.bfloat16).float()
    var = pooled.pow(2).mean(dim=-1, keepdim=True)
    normed = pooled * torch.rsqrt(var + EPS) * rms_weight.float()

    nblk = normed.shape[0]
    # checkpoint: freqs_cis[:cutoff:ratio] -> block b uses position b*ratio
    rope_positions = torch.arange(nblk, dtype=torch.int64) * ratio
    return normed, rope_positions


def apply_repo_rope(vec, cos_sin_cache, pos):
    """The in-repo reference's GPT-J rotation, reused verbatim on both sides."""
    cos, sin = cos_sin_cache[pos].float().chunk(2)
    nope, rope = vec.split([HEAD_DIM - ROPE_DIM, ROPE_DIM])
    rope = torch.stack(
        [rope[0::2] * cos - rope[1::2] * sin,
         rope[1::2] * cos + rope[0::2] * sin], dim=-1).reshape(ROPE_DIM)
    return torch.cat([nope, rope])


# --------------------------------------------------------------------------
def build_state_cache(kv, score_ape, state_block_size, coff):
    """Pack per-token (kv, score+ape) into the paged layout the repo reference
    reads: row = concat(kv[coff*H], score[coff*H]); page = idx // bs."""
    T = kv.shape[0]
    width = coff * HEAD_DIM
    rows = torch.cat([kv, score_ape], dim=-1)               # [T, 2*width]
    pages = (T + state_block_size - 1) // state_block_size
    cache = torch.zeros(pages, state_block_size, 2 * width, dtype=torch.float32)
    for t in range(T):
        cache[t // state_block_size, t % state_block_size] = rows[t]
    block_table = torch.arange(pages, dtype=torch.int32).unsqueeze(0)
    return cache, block_table


def run_case(ratio, nblk, state_block_size, seed, repo_ref, cast_bf16):
    torch.manual_seed(seed)
    coff = 1 + (ratio == 4)
    T = nblk * ratio
    width = coff * HEAD_DIM

    kv = torch.randn(T, width, dtype=torch.float32)
    score = torch.randn(T, width, dtype=torch.float32) * 1.5
    ape = torch.randn(ratio, width, dtype=torch.float32) * 0.5
    rms_weight = torch.rand(HEAD_DIM, dtype=torch.float32) + 0.5
    cos_sin_cache = torch.randn(T + 8, ROPE_DIM, dtype=torch.float32)

    # APE is fused into the stored score in our production path (indexed by the
    # token's own within-block position), which is what the checkpoint does too.
    score_ape = score + ape[torch.arange(T) % ratio]

    cache, block_table = build_state_cache(kv, score_ape, state_block_size, coff)

    # Compression happens on the LAST token of each block: (pos+1) % ratio == 0
    positions = torch.arange(nblk, dtype=torch.int64) * ratio + (ratio - 1)

    got = repo_ref(
        state_cache=cache, block_table=block_table, positions=positions,
        rms_weight=rms_weight, cos_sin_cache=cos_sin_cache,
        compress_ratio=ratio, overlap=int(ratio == 4),
        rms_eps=EPS, return_full_cache=True,
    ).float()

    want_pre, want_pos = ckpt_compress(kv, score_ape, ape * 0, rms_weight,
                                       ratio, cast_bf16)
    # ape already folded into score_ape above; pass zeros so it is not applied
    # twice (the checkpoint adds it once, before overlap_transform).
    want = torch.stack([apply_repo_rope(want_pre[b], cos_sin_cache,
                                        int(want_pos[b]))
                        for b in range(nblk)]).to(torch.bfloat16).float()

    repo_pos = (positions // ratio) * ratio
    pos_match = bool(torch.equal(repo_pos, want_pos))
    denom = want.norm().clamp_min(1e-12)
    rel = ((got - want).norm() / denom).item()
    maxabs = (got - want).abs().max().item()
    return rel, maxabs, pos_match


def main():
    try:
        repo_ref = load_repo_reference()
    except Exception as e:
        print(f"could not load repo reference: {type(e).__name__}: {e}")
        print("mount it:  -v $PWD/vllm/tests/kernels:/kernels:ro")
        return 2

    print("compressor: repo reference vs CHECKPOINT transcription "
          "(bf16-before-norm as the checkpoint does)")
    print(f"{'ratio':>6} {'nblk':>5} {'state_bs':>9} {'rel_err':>10} "
          f"{'max_abs':>10} {'rope_pos':>9}  verdict")
    print("-" * 72)

    ok = True
    for ratio, nblk, bs, seed in (
        (4, 1, 16, 1),      # block 0 only: exercises the -inf/0 overlap pad
        (4, 2, 16, 2),
        (4, 5, 16, 3),
        (4, 5, 4, 4),       # state block boundary == compress ratio
        (4, 9, 8, 5),
        (128, 3, 64, 6),    # non-overlap branch (ratio 128 layers)
    ):
        rel, maxabs, pos_match = run_case(ratio, nblk, bs, seed, repo_ref,
                                          cast_bf16=True)
        # bf16 store on both sides -> ~1e-2 is the rounding floor, not a bug
        good = rel < 5e-2 and pos_match
        ok &= good
        print(f"{ratio:>6} {nblk:>5} {bs:>9} {rel:>10.3e} {maxabs:>10.3e} "
              f"{str(pos_match):>9}  {'MATCH' if good else 'DIVERGE'}")

    print()
    print("RESULT:", "structural semantics AGREE" if ok
          else "STRUCTURAL DIVERGENCE — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
