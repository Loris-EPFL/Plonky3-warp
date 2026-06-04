![Plonky3-powered-by-polygon](https://github.com/Plonky3/Plonky3/assets/86010/7ec356ad-b0f3-4c4c-aa1d-3a151c1065e7)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Plonky3/Plonky3/blob/main/LICENSE-MIT)
[![License](https://img.shields.io/github/license/Plonky3/Plonky3)](https://github.com/Plonky3/Plonky3/blob/main/LICENSE-APACHE)

Plonky3 is a toolkit which provides a set of primitives, such as polynomial commitment schemes, for implementing polynomial IOPs (PIOPs). It is mainly used to power STARK-based zkVMs, though in principle it may be used for PLONK-based circuits or other PIOPs.

This fork adds `p3-warp`, a WARP accumulation crate integrated into the Plonky3 workspace. The `warp/` crate is intended to be used as a toolkit component: it exposes WARP accumulation over Reed-Solomon codewords, root-IOP claim collection, WHIR-backed linear-opening proofs, and finalizer backends that can be plugged into higher-level proving pipelines. The current implementation is research-oriented and benchmarked inside this repository, but its API shape follows the Plonky3 style of reusable protocol building blocks rather than a standalone application.

For questions or discussions, please use the Telegram group, [t.me/plonky3](https://t.me/plonky3).

codex resume 019e239d-ab82-78d2-bab7-8513aa56de22
# 1. Overview

Plonky3-warp is a Rust/no_std-oriented proof-system workspace. The security-critical code is not a web service; it is a set of libraries and benchmarks used to build and verify STARK/WHIR/WARP zero-knowledge or succinct proofs. The main asset is **soundness**: a verifier must not accept a proof for a false circuit/trace/accumulation claim. Secondary assets are binding of proofs to public inputs and circuit shape, availability of verifiers consuming untrusted proofs, and zero-knowledge/hiding when optional hiding commitments or randomized protocols are used.

The highest-risk path is the WHIR-native circuit prover/verifier in `circuit-prover/src/whir_native.rs`, where circuit traces become WHIR oracle commitments, local constraints, read-bus checks, Poseidon2 shift-bus checks, and table openings. WARP composition and finalization live in `warp/src/whir_compiler/`, `warp/src/root_iop/`, `warp/src/finalize/whir/`, `warp/src/protocol/`, and `warp/src/root.rs`. Commitment security depends on `commit/src/`, `merkle-tree/src/`, `whir/src/pcs/`, `whir/src/sumcheck/`, `challenger/src/`, `field/src/`, and `matrix/src/`.

# 2. Threat model, Trust boundaries and assumptions

**Attacker-controlled inputs.** In real deployments a malicious prover can control serialized proof objects (`WhirNativeCircuitProof`, `WarpProof*`, `WhirProof`, Merkle opening proofs), commitments, claimed openings, proof vector lengths, field elements, and private witness values. In a prover-as-a-service setting, submitted circuits/traces may also be attacker-controlled, but normal verifier deployments should treat the circuit/verifier key and protocol parameters as fixed. Public inputs may be user-controlled but are part of the public statement and must be transcript-bound, not secret.

**Operator-controlled inputs.** Protocol parameters (`security_level`, `pow_bits`, `FoldingFactor`, MMCS hash/compressor, `WhirNativeCircuitOptions`), allowed operation sets, circuit definitions, feature flags, and maximum proof sizes are operator policy. If these are accepted from a proof, soundness can collapse; they should be pinned by the verifier.

**Developer-controlled inputs.** Tests, benchmark fixtures, local `warp/benches/sumcheck.rs`, and environment variables are development tooling. Bugs there are usually not exploitable unless copied into production automation.

Assumptions: field arithmetic and SIMD packing are correct; Poseidon2/Blake3/Merkle/hash permutations meet their claimed collision/preimage properties; Fiat-Shamir challengers are cryptographically sound; CSPRNGs used for hiding commitments are properly seeded; and verifiers run with resource limits appropriate for untrusted serialized inputs. Side-channel resistance is not a primary property of these generic arithmetic libraries unless explicitly used in a secret-bearing prover environment.

# 3. Attack surface, mitigations and attacker stories

**WHIR-native circuit proof verification.** `circuit-prover/src/whir_native.rs` is the central attack surface. A malicious prover may try to swap table metadata, use stale proof layouts, omit local constraints, duplicate terminal claims, or substitute read/shift-bus rows. Important controls include recomputing public and shape digests (`compute_public_io_digest`, `compute_shape_digest`), recomputing expected table metadata (`whir_native_expected_table_metadata`), checking `opening_mode`, table counts, `TABLE_LAYOUT_VERSION`, active rows, widths, terminal opening counts, and verifying local sumchecks (`verify_sumcheck`) before WHIR opening verification. Domain-separated transcript contexts such as `observe_table_context`, `observe_circuit_constraint_context`, `observe_read_bus_challenge_context`, and Poseidon2 shift contexts bind public inputs, shape, options, commitments, metadata, and sampled challenges. A false-accept bug here is critical.

**Fiat-Shamir transcript binding.** `challenger/src/`, `warp/src/transcript.rs`, `whir/src/fiat_shamir/`, and the observe functions in `whir_native.rs`/`warp/src/root.rs` decide which public data influences challenges. The attacker story is replaying a valid proof under a different circuit, commitment order, opening mode, table id, or batch layout. The code uses explicit tags/constants, length observations, metadata observations, and commitment observations. Review should confirm every sampled challenge is preceded by all data it is meant to bind, especially before batching/sumcheck reductions.

**WARP root IOP and WHIR compiler boundary.** `warp/src/root_iop/` records commitments and typed claim ids; `warp/src/whir_compiler/` proves those claims with WHIR. Risks are claim reordering, dropped/duplicated claim ids, base/extension confusion, or virtual commitments not being tied to real backend commitments. Mitigations include monotone ids, `RootIopOracleField`, shape checks, `RootIopBoundCommitment::observe_into`, and verifier collection of expected claims before final proof verification. Tests in `warp/src/whir_compiler/tests.rs` cover swapped commitments and tampered virtual evaluations; similar negative tests should accompany changes.

**Commitment/opening binding.** `merkle-tree/src/mmcs.rs` verifies batch size, height compatibility, index bounds, proof sibling counts, cap membership, and opening shapes. `commit/src/` defines PCS/MMCS traits; `whir/src/pcs/` layers WHIR opening proofs on top. Attacker stories include malformed Merkle paths, wrong cap heights, cross-matrix dimension confusion, and commitment substitution. Existing typed errors (`MerkleTreeError`, WHIR verifier errors) are useful, but prover-only `assert!`/`panic!` paths should not be reachable from untrusted verifier inputs.

**Sumcheck and batching reductions.** `circuit-prover/src/whir_native_sumcheck.rs`, `warp/src/sumcheck.rs`, `whir/src/sumcheck/`, and `whir/src/constraints/statement/linear.rs` must validate round counts, polynomial degrees, claimed sums, arities, and final residual openings. Bugs can turn many constraints into one unchecked claim. Review challenge reuse, degree assumptions, and whether claimed terminal points/values are verifier-derived rather than prover-selected.

**Serialization and resource use.** Proof structs derive `serde` across `whir_native.rs`, `warp/src/whir_compiler/types.rs`, `commit/src/mmcs.rs`, and `merkle-tree/src/merkle_tree.rs`. Deserialization alone does not validate semantic lengths. An attacker can send huge vectors or malformed shapes causing memory/CPU DoS before verification returns an error. Production callers should use bounded decoders, maximum proof size/recursion depth, and catch unwinds if exposing verification as a service. Non-canonical field encodings are mostly handled by field deserializers; Goldilocks intentionally accepts any internal `u64`, so byte-level proof canonicality should not be assumed.

**Unsafe/panic surfaces.** `field/src/` and `matrix/src/` contain SIMD/transmute/unchecked-row optimizations; `merkle-tree` uses unchecked matrix rows after internal bounds checks. Memory safety bugs are less likely from proof bytes directly but would be high impact. Panics from `assert!`, `unwrap`, or unchecked dimension assumptions are availability issues unless they skip a verification check.

**Out of scope / lower criticality.** There is no authentication, authorization, sessions, CSRF, XSS, SSRF, SQL injection, or multi-tenant web boundary in this repository. Benchmark/report tooling and environment-variable controls are low risk unless used in production orchestration. Diagnostic trace payloads may reveal witnesses and should remain debugging artifacts.

# 4. Criticality calibration (critical, high, medium, low)

**Critical.** Any bug that lets a verifier accept an invalid proof or accumulator update: missing binding of public inputs/circuit shape/options/commitments into Fiat-Shamir; accepting prover-supplied metadata or opening points instead of recomputing them; Merkle/WHIR opening verification that can be forged; read-bus, Poseidon2 shift-bus, or local constraint checks that can be reordered/dropped/duplicated; field arithmetic/challenger flaws that invalidate soundness assumptions.

**High.** Verification panics or undefined behavior reachable from untrusted proof bytes in a networked verifier; stale proof-layout deserialization accepted for a different mode/version; attacker-controlled protocol parameters reducing security below policy; zero-knowledge randomness misuse in hiding commitments that reveals witness data; significant soundness loss limited to a feature/table type.

**Medium.** Resource-exhaustion through oversized proofs, excessive vector lengths, or expensive malformed inputs; proof malleability/non-canonical encodings that do not enable false acceptance; verifier error paths that are inconsistent but fail closed; side-channel leakage from prover operations when witnesses are secret in a shared environment.

**Low.** Developer-tool and benchmark issues, logging/diagnostic information disclosure in non-production paths, prover-only correctness bugs that merely prevent proof generation, and panics requiring operator-controlled invalid circuits or parameters.

## Status

Fields:
- [x] Mersenne31
  - [x] "complex" extension field
  - [x] ~128 bit extension field
  - [x] AVX2
  - [x] AVX-512
  - [x] NEON
- [x] General 31 bit fields (BabyBear and KoalaBear)
  - [x] ~128 bit extension field
  - [x] AVX2
  - [x] AVX-512
  - [x] NEON
- [x] Goldilocks
  - [x] ~128 bit extension field

Generalized vector commitment schemes
- [x] generalized Merkle tree

Polynomial commitment schemes
- [x] FRI-based PCS
- [ ] tensor PCS
- [ ] univariate-to-multivariate adapter
- [ ] multivariate-to-univariate adapter

PIOPs
- [x] univariate STARK
- [ ] multivariate STARK
- [ ] PLONK

Codes
- [x] Brakedown
- [x] Reed-Solomon

Interpolation
- [x] Barycentric interpolation
- [x] radix-2 DIT FFT
- [x] radix-2 Bowers FFT
- [ ] four-step FFT
- [x] Mersenne circle group FFT

Hashes
- [x] Rescue
- [x] Poseidon
- [x] Poseidon2
- [x] BLAKE3
  - [ ] modifications to tune BLAKE3 for hashing small leaves
- [x] Keccak-256
- [x] SHA-256
- [x] Monolith


## Benchmarks

Many variations are possible, with different fields, hashes, and so forth, which can be controlled through the command line.

For example, to prove 2^20 Poseidon2 permutations of width 16, using the `KoalaBear` field, `Radix2DitParallel` DFT, and `KeccakF` as the Merkle tree hash:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --example prove_prime_field_31 --release --features parallel -- --field koala-bear --objective poseidon-2-permutations --log-trace-length 17 --discrete-fourier-transform radix-2-dit-parallel --merkle-hash keccak-f
```

Currently the options for the command line arguments are:
- `--field` (`-f`): `mersenne-31` or `koala-bear` or `baby-bear`.
- `--objective` (`-o`): `blake-3-permutations, poseidon-2-permutations, keccak-f-permutations`.
- `--log-trace-length` (`-l`): Accepts any integer between `0` and `255`. The number of permutations proven is `trace_length, 8*trace_length` and `trace_length/24` for `blake3, poseidon2` and `keccakf` respectively. 
- `--discrete-fourier-transform` (`-d`): `radix-2-dit-parallel, recursive-dft` or `small-batch-dft`. This option should be omitted if the field choice is `mersenne-31` as the circle stark currently only supports a single discrete fourier transform.
- `--merkle-hash` (`-m`): `poseidon-2, keccak-f`.

Extra speedups may be possible with some configuration changes:
- `JEMALLOC_SYS_WITH_MALLOC_CONF=retain:true,dirty_decay_ms:-1,muzzy_decay_ms:-1` will cause jemalloc to hang on to virtual memory. This may not affect the very first proof much, but can help significantly with subsequent proofs as fewer pages (if any) will need to be newly assigned by the OS. These settings might not be suitable for all production environments, e.g. if the process' virtual memory is limited by `ulimit` or `max_map_count`.
- Adding `lto = "fat"` in the top-level `Cargo.toml` may improve performance slightly, at the cost of longer compilation times.

## CPU features

Plonky3 contains optimizations that rely on newer CPU instructions unavailable in older processors. These instruction sets include x86's [BMI1 and 2](https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set), [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2), and [AVX-512](https://en.wikipedia.org/wiki/AVX-512). Rustc does not emit those instructions by default; they must be explicitly enabled through the `target-feature` compiler option (or implicitly by setting `target-cpu`). To enable all features that are supported on your machine, you can set `target-cpu` to `native`. For example, to run the tests:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test
```

## Known issues

The verifier might panic upon receiving certain invalid proofs.


## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## Guidance for external contributors

Do you feel keen and able to help with Plonky3? That's great! We
encourage external contributions!

We want to make it easy for you to contribute, but at the same time, we
must manage the burden of reviewing external contributions. We are a
small team, and the time we spend reviewing external contributions is
time we are not developing ourselves.

We also want to help you avoid inadvertently duplicating work that
is already underway, or building something that we will not
want to incorporate.

First and foremost, please keep in mind that this is a highly
technical piece of software and contributing is only suitable for
experienced mathematicians, cryptographers, and software engineers.

The Polygon Zero Team reserves the right to accept or reject any
external contribution for any reason, including a simple lack of time
to maintain it (now or in the future); we may even decline to review
something that is not considered a sufficiently high priority for us.

To avoid disappointment, please communicate your intention to
contribute openly, while respecting the limited time and availability
we have to review and provide guidance for external contributions. It
is a good idea to drop a note in our public Discord #development
channel about your intention to work on something, whether an issue, a
new feature, or a performance improvement. This is probably all that's
really required to avoid duplication of work with other contributors.

What follows are some more specific requests for how to write PRs in a
way that will make them easy for us to review. Deviating from these
guidelines may result in your PR being rejected, ignored, or forgotten.


### General guidance for your PR

Obviously, PRs will not be considered unless they pass our Github
CI. The GitHub CI is not executed for PRs from forks, but you can
simulate the GitHub CI by running the commands in
`.github/workflows/ci.yml`.

Under no circumstances should a single PR mix different purposes: Your
PR is either a bug fix, a new feature, or a performance improvement,
never a combination. Nor should you include, for example, two
unrelated performance improvements in one PR. Please just submit
separate PRs. The goal is to make reviewing your PR as simple as
possible, and you should be thinking about how to compose the PR to
minimize the burden on the reviewer.

Plonky3 uses stable Rust, so any PR that depends on unstable features
is likely to be rejected. It's possible that we may relax this policy
in the future, but we aim to minimize the use of unstable features;
please discuss with us before enabling any.

Please do not submit any PRs consisting of merely minor fixes to documentation.
They will not be accepted. If you find something that bothers you particularly,
make an issue and we will fix it when we find the time.

Here are a few specific guidelines for the three main categories of
PRs that we expect:


#### The PR fixes a bug

In the PR description, please clearly but briefly describe

1. the bug (could be a reference to a GH issue; if it is from a
   discussion (on Discord/email/etc. for example), please copy in the
   relevant parts of the discussion);
2. what turned out to cause the bug; and
3. how the PR fixes the bug.

Wherever possible, PRs that fix bugs should include additional tests
that (i) trigger the original bug and (ii) pass after applying the PR.


#### The PR implements a new feature

If you plan to contribute the implementation of a new feature, please
double-check with the Polygon Zero team that it is a sufficient
priority for us that it will be reviewed and integrated.

In the PR description, please clearly but briefly describe

1. what the feature does
2. the approach taken to implement it

All PRs for new features must include a suitable test suite.


#### The PR improves performance

Performance improvements are particularly welcome! Please note that it
can be quite difficult to establish true improvements for the
workloads we care about. To help filter out false positives, the PR
description for a performance improvement must clearly identify

1. the target bottleneck (only one per PR to avoid confusing things!)
2. how performance is measured
3. characteristics of the machine used (CPU, OS, #threads if appropriate)
4. performance before and after the PR


### Licensing

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
