# Concepts

Shared domain vocabulary for this project — entities, named processes, and
status concepts with project-specific meaning. Seeded with core domain
vocabulary, then accretes as ce-compound and ce-compound-refresh process
learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## 2-bit MoE serving (Moet / W2)

### W2
The 2-bit compression tier for routed MoE expert weights: each weight is one
of four sign-symmetric levels, scaled per 32-element group. W2 is the base
tier every expert always has; refinement tiers (FP4-class deltas) may sit on
top of it but never replace it. The name also denotes the GEMM family that
consumes this format.

### Plane
The packed on-device representation of W2 expert weights: fragment-major
2-bit codes plus per-32-group exponent scale bytes, laid out so tensor-core
kernels can load them directly. Planes are produced once by quantization and
are identical for every kernel implementation that reads them.

### Quantization cache
The on-disk artifact holding quantized planes (or packs) so later boots skip
the slow first-run quantization. Its identity is the serving configuration —
tensor-parallel size, Residency, and quantizer settings such as Scale refit —
and a cache from one configuration never applies to another; booting a
mismatched configuration silently triggers a full re-quantization that
overwrites it.

### Residency
Where the W2 base lives during serving: on the GPUs (VRAM-heavy, no
per-step streaming) or in pinned host RAM with a GPU pool (RAM-heavy,
streams over PCIe). Residency is part of the Quantization cache identity.

### Scale refit
A quantization-time option that tests an alternative scale for each weight
group and keeps whichever reproduces the block with less error. It changes
plane content — and therefore the Quantization cache identity — but not the
serving kernels.

### Pair
The unit of MoE GEMM work: one routed expert coupled with one token group.
The dispatcher builds one Descriptor per Pair; a Pair whose token count is
zero is dead and every kernel implementation must leave its output untouched.

### Descriptor
The fixed-layout record that hands one Pair's operand locations, scales,
output location, and live-row count to a kernel. All kernel implementations
consume the same Descriptor table, which is what makes them interchangeable
behind the dispatcher.

### Tier
A dispatch class of the MoE GEMM with its own input envelope and kernel
registry: the decode tier carries small token groups, the prefill tier
carries large ones, and refinement tiers serve delta formats. A kernel
validated for one Tier's envelope must not receive another Tier's calls —
Tier identity, not shape alone, gates dispatch.

### Emulation
The portable Triton implementation of the W2 GEMM. It serves every Tier,
shape, and architecture, and is the correctness backstop: fast paths
activate opportunistically on top of it and fall back to it on any load or
parity failure.

### Parity gate
The boot-time self-test a fast-path kernel must pass before entering the
dispatch registry: a random case is run through the kernel and compared
against a reference implementation on the deployed silicon, per supported
shape. Failure demotes the fast path to the Emulation and never blocks the
engine from serving.
