# Working notes: guppy-registry

Working notes for continued development of this package registry. See [README.md](README.md) for the user-facing pitch. This file is for the next person (human or Claude) picking up the work -- versions pinned, gotchas hit, how correctness was actually verified, and what's planned next.

## Versions used (pinned 2026-08-28)

Installed into `.venv` via `pip install guppylang pytest` (Python 3.14.5, Windows):

| package | version |
|---|---|
| `guppylang` | 1.0.2 |
| `guppylang-internals` | 1.0.2 |
| `hugr` (Python bindings) | 0.18.5 |
| `selene-sim` | 0.3.0 |
| `selene-hugr-qis-compiler` | 0.4.2 |
| `selene-core` | 0.3.0 |
| `pytket` | 2.18.1 |
| `tket` (new Rust tket, separate from pytket) | 0.15.6 |
| `numpy` | 2.5.2 |
| `pytest` | 9.1.1 |

`guppylang` 1.0.2's wheel metadata declares `Requires-Python: <4,>=3.12` and classifiers for 3.12/3.13/3.14 -- Python 3.14 is supported despite being very new.

**Do not assume any of the above from memory or from older guppylang tutorials.** The API changed substantially at the 1.0 release (see below) and this project targets 1.0.2 specifically. Re-verify against the installed version (`pip show guppylang`) and the installed source (`site-packages/guppylang/`) before trusting any pattern below on a future guppylang version.

## Docs consulted

- [Getting Started (v1.0.1)](https://docs.quantinuum.com/guppy/getting_started.html) -- install, basic `@guppy` usage, `.emulator()`.
- [Canonical QPE example (v1.0.1)](https://docs.quantinuum.com/guppy/guppylang/examples/canonical-qpe.html) -- source of the `qft`/`swap` pattern used in this package (comptime QFT + `with dagger:` for phase estimation).
- [state_output / Debugging with state_result (v1.0.1)](https://docs.quantinuum.com/guppy/guppylang/examples/state_results.html) -- `state_output`, `.emulator().statevector_sim()`, `partial_state_dicts()`.
- [guppylang.defs API](https://docs.quantinuum.com/guppy/api/defs.html) -- `.compile()`/`.compile_function()`/`.emulator(libs=...)` signatures.
- [guppylang.emulator API](https://docs.quantinuum.com/guppy/api/emulator.html) -- `EmulatorBuilder`/`EmulatorInstance`/`EmulatorResult` methods.
- [Selene simulation backends](https://docs.quantinuum.com/selene/user_guide/simulation.html) -- Stim (Clifford-only) vs Quest (general statevector).
- [Quantinuum/guppylang on GitHub](https://github.com/Quantinuum/guppylang).

The docs above are pinned to v1.0.1; the installed version is 1.0.2. Almost everything matched exactly, with the exceptions logged below -- always cross-check against installed source (`site-packages/guppylang/`) when something doesn't work as documented, rather than assuming the doc or your memory is right.

## API gotchas hit during development (guppylang 1.0.2)

These cost real debugging time. Read before writing a new package.

1. **`state_result` -> `state_output`.** Renamed in the 1.0 line; `guppylang.std.debug.state_result` still exists as a deprecated alias (`state_result = state_output`), but write `state_output`.

2. **`link_name` kwarg removed from `@guppy`.** `@guppy(link_name="...")` now raises `TypeError`. Use the separate `@link_name(...)` decorator from `guppylang.library`, placed *below* `@guppy`/`@guppy.declare`:
   ```python
   from guppylang import guppy
   from guppylang.library import link_name

   @guppy
   @link_name("my_registry.my_func")
   def my_func(...) -> None: ...
   ```

3. **`.compile()`/`.emulator()`/`.compile_function()` default to `OptimizationLevel.Default`, which imports `pytket.passes.RemoveRedundancies`.** On this dev machine, a Windows Application Control policy blocks pytket's compiled extension (`pytket._tket.architecture`, a `.pyd`) outright -- `ImportError: DLL load failed while importing architecture: An Application Control policy has blocked this file`, reproducible on every attempt (not transient). Fix: use `.with_minimal_opt()` (applies zero passes, never imports pytket) or `.with_opt_level(OptimizationLevel.Classical)` (uses the newer Rust `tket` package instead of `pytket`, which is *not* blocked). This whole package's source and tests use `.with_minimal_opt()` throughout for this reason -- also arguably the right default for a small library function, since it keeps the compiled circuit structurally predictable.

   Separately: on this machine, *other* native DLLs/executables (`hugr._hugr`, and the actual Selene runner subprocess spawned by `emulator(...).run()`) are sometimes transiently blocked by the same Application Control policy the *first* time a fresh process touches them, then succeed on an immediate retry -- looks like a scan-on-first-access policy, not a real block. `tests/test_correctness.py`'s `_run_case()` retries subprocess launches up to 3 times for this reason. If you hit `OSError: [WinError 4551] An Application Control policy has blocked this file` or a DLL `ImportError` mentioning "Application Control policy", retry once or twice before assuming something is actually broken.

4. **Linking a compiled function's `Package` via `libs=[...]` fails unless the module's entrypoint is reset to its root.** `some_func.compile_function()` still marks that specific function as the HUGR module's entrypoint. Linking two modules that both have a non-root entrypoint raises `hugr._hugr.linking.HugrLinkingError: Cannot link two modules with non-root entrypoints together`. Fix, before serializing/linking a library package:
   ```python
   package = my_func.with_minimal_opt().compile_function()
   for module in package.modules:
       module.entrypoint = module.module_root
   ```
   See `packages/qft/examples/build_lib.py` for the full working pattern (build + `Package.to_bytes()`/`from_bytes()` round trip + `@guppy.declare` + `@link_name` + `.emulator(libs=[...])`), validated end-to-end across two separate Python processes.

5. **`iqft` run in isolation (no call to `qft` anywhere in the same compiled program) produces a different, wrong unitary than the exact same `iqft` run in a program that also calls `qft`, and this can non-deterministically corrupt other, unrelated compilations in the same process too. Significant, unresolved upstream.**

   Core (deterministic) repro: compile+run a program that only calls `iqft` on X-prepared basis states and reads out `state_output` -- the resulting matrix is unitary but does not match the exact inverse-QFT matrix (`numpy.fft.fft(eye, norm="ortho")`), off by a consistent, non-trivial pattern (extra ~`pi/2**(n+1)`-scale phase structure), reproduced identically across many runs. The *exact same* `iqft` function, in a program that also calls `qft` on the *same register* (either order: `qft` then `iqft`, or `iqft` then `qft`; a throwaway `qft` call on an *unrelated* register does **not** fix it -- only a `qft` call on the same register does) reproduces the true inverse-QFT matrix to ~1e-16. This held both for an `iqft` implemented via `with dagger: qft(qs)` and for a fully hand-written, dagger-free `iqft` with the swaps/rotations explicitly reversed and negated (rules out `dagger` specifically), and is independent of whether the driver is `@guppy` or `@guppy.comptime` (rules out comptime-calling-comptime as a separate cause -- an earlier version of this file wrongly claimed that as a distinct bug; it isn't, see below).

   Broader, non-deterministic symptom: a pytest file with `test_qft_matches_numpy_ifft` (sweeping `qft` alone over n=1..4) *and* `test_iqft_matches_numpy_fft` (sweeping `iqft` alone over n=1..4) produces failures whose exact pattern varies run to run -- one run failed only `qft[4]` and `iqft[2,3,4]`; a later run of the identical file failed `qft[1]` and `iqft[1,2,3,4]`. Critically, **the same `qft`-only sweep, in a file with no `iqft` at all, passed cleanly 3/3 times** run via actual pytest. So the corruption isn't simply "the same generic function monomorphized for multiple n" (an earlier version of this file claimed that as a *second*, separate bug -- it also isn't; a standalone script sweeping only `qft` over n=1,2,3,4 in one process was clean every time). It specifically requires `iqft` (or something about how it's compiled) to be present/exercised, and once triggered, the corruption isn't confined to `iqft`'s own results.

   We were unable to isolate a smaller repro (e.g. a single-qubit `rz`/`crz` case) or find the root cause in the time available. **Open item for whoever picks this up next**: try to reduce this to a minimal repro (ideally without depending on pytest specifically, to nail down whether pytest's import/collection machinery is actually relevant or just a red herring) and file it upstream against `guppylang`/`hugr`. Practical consequence for this package: `iqft`'s correctness is verified via round trips against `qft` (see `packages/qft/tests/test_correctness.py`), each test case run in its own subprocess (sidesteps the cross-contamination entirely, deterministic or not); a test documenting the isolated-failure case is kept and marked `xfail(strict=True)` rather than deleted.

6. **A plain Python function (or closure) cannot be called indirectly from inside a `@guppy` function body**, even if it just forwards to a real guppy definition: `guppy` resolves calls statically against known guppy globals, not arbitrary Python values. `def helper(q): real_guppy_func(q)` called as `helper(q)` inside a `@guppy` body raises `GuppyError` (`UnsupportedPythonValueError`). This ruled out a `_basis_state_matrix(n, op)`-style shared test helper; `qft`/`iqft` test drivers in this package are written out twice instead.

7. **Import statements are not allowed inside a `@guppy` function body** (only at module scope) -- raises `GuppyError` (`UnsupportedError`, "This statement").

8. `array`, `output`, `control`, `dagger`, `nat`, `owned`, `py`, `comptime`, etc. are all re-exported from `guppylang.std.builtins` (see `site-packages/guppylang/std/builtins.py`) -- no need to import from their "real" submodules (`guppylang.std.array`, `guppylang.std.lang`, `guppylang.std.num`, ...) directly.

9. `guppylang.std.num.int`/`nat` support `>>`, `&`, `//`, `%`, etc. -- real bitwise/integer ops are available in plain (non-comptime) guppy code, not just in `@guppy.comptime`. Used for basis-state preparation loops (`(k >> i) & 1`) in this package's tests without needing `comptime`.

10. `angle` (from `guppylang.std.angles`) is a frozen struct wrapping `halfturns: float`, *not* radians directly; `pi = angle(1)`. `float(some_angle) == some_angle.halfturns * pi`. Arithmetic like `pi / 2.0 ** k` works via `angle.__truediv__`.

11. `crz(control, target, angle)` -- control qubit first, target second (`Qubit ordering: [control, target]` per its docstring); matches the official QPE doc example.

12. **`with control(q0, q1): z(target)` -- a multi-controlled Z built from 2+ control qubits via the `control` modifier -- is broken in guppylang 1.0.2.** Found while building `packages/grover`. Confirmed with 2 controls (CCZ) and 3 controls (CCCZ): applying it to a uniform superposition produces a matrix that stays unitary but is measurably wrong (not the expected phase-flip-on-the-all-ones-state unitary) -- e.g. for 2 controls, `CCZ|+++>` should be `[0.3536]*7 + [-0.3536]` (real), but the actual result has spurious *imaginary* components on the two highest-index amplitudes instead: `[0.3536]*6 + [-0.3536j, 0.3536j]`. **`with control(q0, q1): x(target)` (multi-controlled X) is correct** -- it matches the built-in `toffoli` gate exactly, for both 2 and 3 controls. Workaround, the standard `H . (multi-controlled X) . H = multi-controlled Z` identity, confirmed exact (~1e-16) for 2 and 3 controls:
    ```python
    h(target)
    with control(*controls):
        x(target)
    h(target)
    ```
    `packages/grover/src/grover/grover.py` uses this throughout instead of a direct multi-controlled `z`. Not yet filed upstream; a minimal repro is in `packages/grover/tests/test_correctness.py::test_direct_multi_controlled_z_is_broken` (kept as `xfail(strict=True)`).

13. **The `control`/`dagger`/`power` `with`-modifiers are explicitly disallowed inside `@guppy.comptime` function bodies** -- calling them there raises `GuppyComptimeError` at runtime (their Python stub implementations in `guppylang/std/lang.py` just raise unconditionally; the real modifier logic only exists for the plain-`@guppy` AST path). This isn't documented anywhere we found -- discovered by trying it. Any function that needs `with control(...):` (e.g. a multi-controlled gate) must be a plain `@guppy` function, not `@guppy.comptime`, even if it's otherwise a good fit for comptime's generic-loop-unrolling style (like `packages/qft`'s `qft`/`iqft`).

14. **`with control(...)` accepts multiple qubits** (`with control(q0, q1, q2): x(target)`), confirmed via `guppylang_internals/cfg/builder.py` (`Control(e, e.args)`, no arity limit beyond "at least 1") and empirically. Not shown anywhere in the docs we found (the only documented example, in the canonical QPE page, uses a single control qubit) -- worth knowing this generalizes.

15. **Explicit generic-argument syntax at a call site, `func[arg1, arg2](...)`, works** for a plain `@guppy` function generic over `nat` parameters that don't appear in any argument's type (so can't be inferred) -- e.g. `oracle[marked](qs)` where `marked: nat` doesn't affect `qs`'s type `array[qubit, 3]`. This is not the same situation as `qft[n]`, where `n` **is** inferred from the array length and explicit instantiation is never needed. The generic argument expression itself must be a literal or a closed-over Python constant -- a live Python function call (`grover_search[marked, optimal_iterations(8, 1)]`) fails with `Error: Invalid type argument (Not a valid type argument)`, because guppy parses that subscript as guppy source to be resolved against guppy's own type system, not as a Python expression to evaluate first. Compute it in Python beforehand and pass the resulting int in via a closed-over variable instead.

## Testing methodology (read before adding a new package)

- **Non-Clifford circuits need `statevector_sim()` (Quest), not `stabilizer_sim()` (Stim).** Stim only simulates Clifford circuits exactly; QFT's controlled phase rotations `crz(..., pi/2**k)` for `k >= 2` are outside the Clifford group, so this package uses Quest throughout. If a future package (e.g. a stabilizer-code primitive) *is* Clifford-only, prefer Stim there -- it's much faster and validates a different thing (that the circuit stays in the stabilizer formalism).
- **Reference implementation**: for QFT, the ground truth is `numpy.fft.ifft`/`fft` with `norm="ortho"` -- the unitary QFT matrix in the `qs[0]` = most-significant-qubit convention this package uses is *exactly* `numpy.fft.ifft(np.eye(2**n), norm="ortho")`, confirmed to ~1e-16 for n=1..4. Establish the qubit-ordering convention empirically first (see the `smoke_convention`-style check: apply `x` to a single qubit, run, and see which output index lights up) rather than assuming a convention from the source code alone.
- **Full-matrix testing over shot sampling**: rather than sampling measurement outcomes, this package's tests reconstruct the full 2^n x 2^n unitary by running every computational basis state through the circuit in one shot (`state_output` per basis state, `partial_states()` to read them back in order) and comparing the resulting matrix directly to the reference. This is exact (no statistical noise/tolerance needed beyond floating point), and also directly checks unitarity as a sanity property.
- **Isolate generic-parameter sweeps and independent circuits into separate subprocesses -- but only when the package's functions actually risk it.** See gotcha #5 above for why `packages/qft` needs this (`packages/qft/tests/_case_runner.py` + `subprocess.run` is the pattern). `packages/grover` doesn't use it: neither `oracle` nor `diffuser` is `@guppy.comptime` (can't be -- see gotcha #13), and we explicitly checked that sweeping `oracle`/`grover_search` over all 8 values of `marked`, and separately over 6 different `iterations` values, in one ordinary pytest process, produces no corruption (repeated runs, always clean). The bug's precondition (comptime + a dagger/derived-function relationship) genuinely doesn't apply here, not just "we didn't happen to hit it" -- see the grover section below. Don't reach for subprocess isolation reflexively; check whether the specific pattern that triggers gotcha #5 is actually present first.
- **Hand-verify at least one small case** in closed form in a test/comment (see `test_qft_hand_verified_n2` in `packages/qft/tests/test_correctness.py`, `test_oracle_flips_only_marked_amplitude` and the diffuser-fixes-`|s>`-exactly test in `packages/grover/tests/test_correctness.py`) so the automated reference formula itself has an independent sanity check.
- **Real-valued circuits (H, X, Z-type gates, no complex phases) still need phase-aligned comparisons against a hand-derived reference.** `packages/grover`'s full `grover_search` circuit's overall statevector sign, for a fixed iteration count, empirically varies with which item is marked (a physically meaningless global phase, not a bug -- see `packages/grover/tests/test_correctness.py`'s `_phase_aligned` and its module docstring). Compare `|amplitude|**2` directly where you can (no alignment needed, and it's what's actually observable), or align signs (flip the reference if `vdot(actual, reference).real < 0`) before comparing raw amplitudes.

## Grover package notes (guppylang 1.0.2, added 2026-08-29)

- No official Grover example was found in Quantinuum's docs (checked `docs.quantinuum.com/guppy` and web search) -- the multi-qubit `control` modifier's behavior (gotchas #12-14 above) was reverse-engineered from `guppylang_internals/cfg/builder.py` and confirmed empirically, not documented anywhere we found. If Quantinuum publishes a Grover/oracle example later, cross-check this package against it.
- **Checked all three of `packages/qft`'s documented bug patterns against this package, as follow-up work explicitly asked us to. None apply, and we verified each rather than assuming:**
  - Gotcha #5 (iqft-isolation / cross-contamination) requires a `@guppy.comptime` function used *only* via composition with another comptime-generic function derived from it (dagger or hand-reversed). Grover has no such relationship -- `oracle` and `diffuser` are independent, non-comptime, non-dagger functions (comptime is actually *unavailable* here, see gotcha #13) -- and we confirmed no corruption sweeping `marked` (8 values) or `iterations` (6 values) in-process.
  - The "comptime calling comptime" framing (folded into gotcha #5 after we found it was a mis-attribution) doesn't apply either, for the same reason: nothing here is `@guppy.comptime`.
  - No multi-`n` monomorphization sweep applies -- this package fixes the register size at 3 qubits (see "What's here" in `packages/grover/README.md` for why full `n`-generality isn't offered).
- **What Grover *did* find**: a new, unrelated bug -- `with control(q0, q1): z(target)` is broken (gotcha #12). Not a QFT-pattern regression; a different corner of the same `control`/modifier machinery (multi-qubit `with control(...)`, gotcha #14, is itself new-to-this-package territory -- `packages/qft` only ever used single-qubit `with control(...)` indirectly via `with dagger:`, and never used `control` directly at all).

## Planned next packages

- ~~**Grover's search**~~ -- done, see `packages/grover`.
- **QAOA / MaxCut ansatz** (next up) -- variational, parametrized-circuit style (distinct from QFT's fixed unitary and Grover's fixed-iteration-count oracle/diffuser), representative of near-term NISQ-style algorithms; would exercise classical-quantum parameter passing (rotation angles as real function arguments, not `nat` generics) and, likely, a classical optimization loop calling back into the compiled circuit repeatedly.
- **VQE for H2** -- another variational algorithm, but chemistry-flavored; would need a Hamiltonian-encoding convention and expectation-value estimation, both new territory for this registry.
- **Basic repetition-code error-correction primitive** -- distinct algorithm family (QEC rather than an algorithm subroutine); Clifford-only, so it's the first candidate suited to Selene's Stim backend rather than Quest, and a good test of whether this registry's testing methodology (full-matrix / reference-implementation comparison) generalizes to a code-distance-parametrized correctness check (e.g. logical error rate under a noise model) rather than a fixed unitary.
