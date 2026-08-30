# guppy-registry

A curated, tested package registry for [guppylang](https://github.com/CQCL/guppylang)/HUGR: quantum algorithm building blocks, distributed as linkable [HUGR Packages](https://docs.quantinuum.com/guppy/api/defs.html).

## Why

HUGR (the compiler IR guppylang targets) already supports linking precompiled Packages into another program -- `guppylang.defs`'s `.emulator(libs: list[Package])` argument exists precisely for this. What's missing is an ecosystem of shared, genuinely-tested packages to link against: today, every guppylang project reimplements the same textbook building blocks (QFT, Grover, ansatze, ...) from scratch, at whatever quality bar that project's own tests demand.

This repo is a first step toward that ecosystem: a home for well-tested, documented guppylang implementations of standard algorithm components, each distributed both as importable guppy source (for projects that want to build on it in Python) and as a standalone compiled `.hugr` file (for projects that just want to link the compiled artifact via `libs=[...]`, no Python dependency on this repo at all).

## What's in it

- **[`packages/qft`](packages/qft/)** -- Quantum Fourier Transform and inverse-QFT. See its [README](packages/qft/README.md) for the API, and its [tests](packages/qft/tests/test_correctness.py) for how correctness is verified (against `numpy.fft`-derived reference matrices, run on Selene's statevector emulator).
- **[`packages/grover`](packages/grover/)** -- Grover's search algorithm (oracle-based amplitude amplification) on a 3-qubit, 1-marked-item register. See its [README](packages/grover/README.md) for the API, and its [tests](packages/grover/tests/test_correctness.py) for how correctness is verified (against the closed-form amplitude-amplification formula, run on Selene's statevector emulator).
- **[`packages/qaoa`](packages/qaoa/)** -- QAOA for MaxCut on a 5-node cycle graph, with a real classical-quantum optimization loop (`scipy.optimize` driving repeated circuit compiles + real shot sampling). See its [README](packages/qaoa/README.md) for the API, and its [tests](packages/qaoa/tests/test_correctness.py) for how correctness is verified (against an exact `scipy.linalg.expm` reference, and against the measured cut distribution actually improving as the circuit gets more layers).
- **[`packages/vqe_h2`](packages/vqe_h2/)** -- VQE for the H2 molecule ground-state energy (4-qubit, Jordan-Wigner-mapped, STO-3G Hamiltonian cited from the literature), with a real classical-quantum optimization loop. See its [README](packages/vqe_h2/README.md) for the API and citations, and its [tests](packages/vqe_h2/tests/test_correctness.py) for how correctness is verified -- this package's rigor bar is higher than the others' (VQE/chemistry is Quantinuum's own flagship application area): every numeric claim is checked against an independent, from-scratch numpy reference, not just the guppy code's own internal consistency.
- **[`packages/repetition_code`](packages/repetition_code/)** -- the 3-qubit bit-flip repetition code (encode / ancilla-based syndrome extraction / classically-controlled correction) -- the textbook first example of quantum error correction, and the first package in this registry whose circuit is genuinely Clifford-only, so it runs on Selene's Stim (stabilizer) backend rather than Quest. See its [README](packages/repetition_code/README.md) for the API, and its [tests](packages/repetition_code/tests/test_correctness.py) for how correctness is verified -- including that syndrome extraction provably doesn't collapse a genuine logical superposition, checked via the exact statevector, not just that classical bit values come out right.
- **[`packages/grover_multi`](packages/grover_multi/)** -- Grover's search with a genuinely multi-item oracle (marking 2 or 3 items out of the same 8-item register `packages/grover` uses), a second-round addition built on `packages/grover`'s gotcha #12 workaround. See its [README](packages/grover_multi/README.md) for the API and the "why a separate package" rationale, and its [tests](packages/grover_multi/tests/test_correctness.py) for how correctness is verified -- against the closed-form multi-item amplitude-amplification formula across several marking patterns and a full iteration-count sweep, run on Selene's statevector emulator.
- **[`packages/vqe_h2_stretched`](packages/vqe_h2_stretched/)** -- VQE for H2 at a stretched, non-equilibrium bond length (2.1A vs. `packages/vqe_h2`'s ~0.742A equilibrium), a second-round addition answering whether `vqe_h2`'s local-minima optimization-landscape finding is H2-equilibrium-specific or general to the ansatz style (it generalizes, with a quantitatively different, also-characterized texture). See its [README](packages/vqe_h2_stretched/README.md) for the Hamiltonian citation and the finding, and its [tests](packages/vqe_h2_stretched/tests/test_correctness.py) for how correctness is verified -- same higher rigor bar as `vqe_h2`.
- **[`packages/repetition_code_v2`](packages/repetition_code_v2/)** -- the bit-flip repetition code generalized to arbitrary (odd) code distance and repeated syndrome-extraction rounds, a second-round addition extending `packages/repetition_code`'s fixed distance-3, single-round primitive to distance-5/7 (minimum-weight/majority-vote decoding, correcting up to `(n-1)//2` simultaneous errors) and continuous multi-round correction. See its [README](packages/repetition_code_v2/README.md) for the API and the generic-over-code-distance finding, and its [tests](packages/repetition_code_v2/tests/test_correctness.py) for how correctness is verified -- including exact backward compatibility with `repetition_code`'s distance-3 behavior and survival across multiple rounds with fresh errors injected between them.

This completes the registry's initial planned set of five packages, plus three second-round additions. More are likely to follow -- see [CLAUDE.md](CLAUDE.md) for ideas and the full rationale/history behind each package so far.

## Structure

```
guppy-registry/
  packages/
    qft/
      src/qft/       guppy source (the package itself)
      tests/          correctness tests, run on Selene's emulator
      examples/       worked examples, incl. compiling to a .hugr file and linking it
      pyproject.toml
      README.md
    grover/           same layout as qft/
    qaoa/             same layout as qft/
    vqe_h2/           same layout as qft/
    repetition_code/  same layout as qft/
    grover_multi/     same layout as qft/
    vqe_h2_stretched/ same layout as qft/
    repetition_code_v2/ same layout as qft/
    <future packages follow the same layout>
  README.md           this file
  CLAUDE.md            versions pinned, gotchas hit, testing methodology, roadmap
```

Each package under `packages/` is independent: its own `pyproject.toml`, its own tests, its own README. There's no shared root package -- `packages/qft`, `packages/grover`, `packages/qaoa`, `packages/vqe_h2`, `packages/repetition_code`, `packages/grover_multi`, `packages/vqe_h2_stretched`, `packages/repetition_code_v2` (and future packages) are meant to be installed and consumed individually.

## Install and use a package

Each package is a standard Python package with `guppylang` as a dependency. To use `qft` in your own project:

```
pip install -e path/to/guppy-registry/packages/qft
```

then, from your own guppy code:

```python
from qft import qft, iqft
```

Same pattern for `grover`: `pip install -e path/to/guppy-registry/packages/grover`, then `from grover import grover_search, oracle, diffuser`. And for `qaoa`: `pip install -e path/to/guppy-registry/packages/qaoa`, then `from qaoa import optimize_qaoa, run_qaoa`. And for `vqe_h2`: `pip install -e path/to/guppy-registry/packages/vqe_h2`, then `from vqe_h2 import optimize_vqe, estimate_energy`. And for `repetition_code`: `pip install -e path/to/guppy-registry/packages/repetition_code`, then `from repetition_code import encode, extract_syndrome, correct`. And for `grover_multi`: `pip install -e path/to/guppy-registry/packages/grover_multi`, then `from grover_multi import grover_2items, grover_3items, optimal_iterations`. And for `vqe_h2_stretched`: `pip install -e path/to/guppy-registry/packages/vqe_h2_stretched`, then `from vqe_h2_stretched import optimize_vqe, estimate_energy`. And for `repetition_code_v2`: `pip install -e path/to/guppy-registry/packages/repetition_code_v2`, then `from repetition_code_v2 import encode, extract_syndrome, correct, correct_for_rounds`.

See [`packages/qft/README.md`](packages/qft/README.md), [`packages/grover/README.md`](packages/grover/README.md), [`packages/qaoa/README.md`](packages/qaoa/README.md), [`packages/vqe_h2/README.md`](packages/vqe_h2/README.md), [`packages/repetition_code/README.md`](packages/repetition_code/README.md), [`packages/grover_multi/README.md`](packages/grover_multi/README.md), [`packages/vqe_h2_stretched/README.md`](packages/vqe_h2_stretched/README.md), or [`packages/repetition_code_v2/README.md`](packages/repetition_code_v2/README.md) for the full API of each, and for how to link the precompiled `.hugr` package instead of depending on this repo's Python source.

## Development setup

Requires Python 3.12-3.14 (see [CLAUDE.md](CLAUDE.md) for the exact versions this repo was developed against).

```
python -m venv .venv
.venv/Scripts/activate   # or: source .venv/bin/activate on macOS/Linux
pip install -e "packages/qft[test]"
pip install -e "packages/grover[test]"
pip install -e "packages/qaoa[test]"
pip install -e "packages/vqe_h2[test]"
pip install -e "packages/repetition_code[test]"
pip install -e "packages/grover_multi[test]"
pip install -e "packages/vqe_h2_stretched[test]"
pip install -e "packages/repetition_code_v2[test]"
```

## Run the tests

```
pytest packages/qft/tests -v
pytest packages/grover/tests -v
pytest packages/qaoa/tests -v
pytest packages/vqe_h2/tests -v
pytest packages/repetition_code/tests -v
pytest packages/grover_multi/tests -v
pytest packages/vqe_h2_stretched/tests -v
pytest packages/repetition_code_v2/tests -v
```

Tests compile and run real guppy circuits against Selene's Quest (statevector) or, for `repetition_code`/`repetition_code_v2`, Stim (stabilizer) emulator, and check the results against independently-computed reference values (see [CLAUDE.md](CLAUDE.md) for the full methodology, including why some `qft` tests deliberately run in subprocesses, and why `repetition_code`/`repetition_code_v2` are the packages that use Stim). Expect ~1-2 minutes per package for `qft`/`grover`/`repetition_code`/`grover_multi`/`repetition_code_v2`, a bit longer for `qaoa`, and longer still for `vqe_h2`/`vqe_h2_stretched` (several live optimization-loop tests each, and by far the highest circuit-compile volume in this registry -- see their READMEs and CLAUDE.md if tests there flake with an `Application Control policy` `OSError`, an environmental issue unrelated to correctness).
