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

5. **Comptime cross-instantiation bug (significant, unresolved upstream): monomorphizing the same `@guppy.comptime` generic function for more than one value of a `nat` type parameter within a single Python process silently corrupts every monomorphization after the first.** Confirmed with `qft[n]`/`iqft[n]` (generic over register size `n`): running `n=1,2,3,4` in sequence within one process gives correct results for the *first* n tried in a given process, and increasingly wrong-but-still-unitary results for subsequent different n values -- no exception, no warning, just a wrong answer that still passes a unitarity check. **Workaround used throughout this package's tests: one (n, ...) case per subprocess** (`packages/qft/tests/_case_runner.py`, invoked via `subprocess.run` from `test_correctness.py`). This is also just a good general habit for guppy-registry test suites: if a package exposes generic functions and tests sweep the generic parameter, isolate each parameter value in its own process rather than trusting in-process reuse.

6. **Calling a `@guppy.comptime` generic function from *inside another* `@guppy.comptime` function body produces silently wrong results.** E.g. a `@guppy.comptime def main(): ... qft(qs) ...` driver gave wrong results, while the exact same driver written as a plain `@guppy def main(): ...` (still calling the comptime `qft`) gave exact results. **Always call comptime-generic package functions (`qft`, `iqft`) from a plain `@guppy` driver, never from another `@guppy.comptime` function.** All tests/examples in this package follow this rule.

7. **`iqft` run in isolation (no call to `qft` anywhere in the same compiled program) produces a different, wrong unitary than the exact same `iqft` run in a program that also calls `qft`.** Isolated repro: compile+run a program that only calls `iqft` on X-prepared basis states and reads out `state_output` -- the resulting matrix is unitary but does not match the exact inverse-QFT matrix (`numpy.fft.fft(eye, norm="ortho")`), off by a consistent, non-trivial pattern (extra ~`pi/2**(n+1)`-scale phase structure). The *exact same* `iqft` function, in a program that also calls `qft` on the same register (either order: `qft` then `iqft`, or `iqft` then `qft`; even a throwaway `qft` call on an *unrelated* register did not reliably fix it in testing -- only a `qft` call on the *same* register did) reproduces the true inverse-QFT matrix to ~1e-16. This held both for an `iqft` implemented via `with dagger: qft(qs)` and for a fully hand-written, dagger-free `iqft` with the swaps/rotations explicitly reversed and negated -- ruling out `dagger` specifically as the cause. We were unable to isolate a smaller repro (e.g. a single-qubit `rz`/`crz` case) or find the root cause in the time available. **Open item for whoever picks this up next**: try to reduce this to a minimal repro and file it upstream against `guppylang`/`hugr`. Practical consequence for this package: `iqft`'s correctness is verified via round trips against `qft` (see `packages/qft/tests/test_correctness.py`), not via an absolute reference matrix in isolation; a test documenting the isolated-failure case is kept and marked `xfail(strict=True)` rather than deleted.

8. **A plain Python function (or closure) cannot be called indirectly from inside a `@guppy` function body**, even if it just forwards to a real guppy definition: `guppy` resolves calls statically against known guppy globals, not arbitrary Python values. `def helper(q): real_guppy_func(q)` called as `helper(q)` inside a `@guppy` body raises `GuppyError` (`UnsupportedPythonValueError`). This ruled out a `_basis_state_matrix(n, op)`-style shared test helper; `qft`/`iqft` test drivers in this package are written out twice instead.

9. **Import statements are not allowed inside a `@guppy` function body** (only at module scope) -- raises `GuppyError` (`UnsupportedError`, "This statement").

10. `array`, `output`, `control`, `dagger`, `nat`, `owned`, `py`, `comptime`, etc. are all re-exported from `guppylang.std.builtins` (see `site-packages/guppylang/std/builtins.py`) -- no need to import from their "real" submodules (`guppylang.std.array`, `guppylang.std.lang`, `guppylang.std.num`, ...) directly.

11. `guppylang.std.num.int`/`nat` support `>>`, `&`, `//`, `%`, etc. -- real bitwise/integer ops are available in plain (non-comptime) guppy code, not just in `@guppy.comptime`. Used for basis-state preparation loops (`(k >> i) & 1`) in this package's tests without needing `comptime`.

12. `angle` (from `guppylang.std.angles`) is a frozen struct wrapping `halfturns: float`, *not* radians directly; `pi = angle(1)`. `float(some_angle) == some_angle.halfturns * pi`. Arithmetic like `pi / 2.0 ** k` works via `angle.__truediv__`.

13. `crz(control, target, angle)` -- control qubit first, target second (`Qubit ordering: [control, target]` per its docstring); matches the official QPE doc example.

## Testing methodology (read before adding a new package)

- **Non-Clifford circuits need `statevector_sim()` (Quest), not `stabilizer_sim()` (Stim).** Stim only simulates Clifford circuits exactly; QFT's controlled phase rotations `crz(..., pi/2**k)` for `k >= 2` are outside the Clifford group, so this package uses Quest throughout. If a future package (e.g. a stabilizer-code primitive) *is* Clifford-only, prefer Stim there -- it's much faster and validates a different thing (that the circuit stays in the stabilizer formalism).
- **Reference implementation**: for QFT, the ground truth is `numpy.fft.ifft`/`fft` with `norm="ortho"` -- the unitary QFT matrix in the `qs[0]` = most-significant-qubit convention this package uses is *exactly* `numpy.fft.ifft(np.eye(2**n), norm="ortho")`, confirmed to ~1e-16 for n=1..4. Establish the qubit-ordering convention empirically first (see the `smoke_convention`-style check: apply `x` to a single qubit, run, and see which output index lights up) rather than assuming a convention from the source code alone.
- **Full-matrix testing over shot sampling**: rather than sampling measurement outcomes, this package's tests reconstruct the full 2^n x 2^n unitary by running every computational basis state through the circuit in one shot (`state_output` per basis state, `partial_states()` to read them back in order) and comparing the resulting matrix directly to the reference. This is exact (no statistical noise/tolerance needed beyond floating point), and also directly checks unitarity as a sanity property.
- **Isolate generic-parameter sweeps and independent circuits into separate subprocesses** -- see gotchas #5 and #7 above. `packages/qft/tests/_case_runner.py` + `subprocess.run` is the pattern; reuse it for future packages that expose `@guppy.comptime` generics.
- **Hand-verify at least one small case** in closed form in a test/comment (see `test_qft_hand_verified_n2` in `packages/qft/tests/test_correctness.py`) so the automated reference formula itself has an independent sanity check.

## Planned next packages

- **Grover's search** -- the other canonical textbook quantum algorithm alongside QFT/QPE; a natural pairing that exercises oracle composition and amplitude amplification, distinct circuit-building patterns from QFT's phase-rotation style.
- **QAOA / MaxCut ansatz** -- variational, parametrized-circuit style (distinct from QFT's fixed unitary), representative of near-term NISQ-style algorithms; would exercise classical-quantum parameter passing.
- **VQE for H2** -- another variational algorithm, but chemistry-flavored; would need a Hamiltonian-encoding convention and expectation-value estimation, both new territory for this registry.
- **Basic repetition-code error-correction primitive** -- distinct algorithm family (QEC rather than an algorithm subroutine); Clifford-only, so it's the first candidate suited to Selene's Stim backend rather than Quest, and a good test of whether this registry's testing methodology (full-matrix / reference-implementation comparison) generalizes to a code-distance-parametrized correctness check (e.g. logical error rate under a noise model) rather than a fixed unitary.
