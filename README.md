# guppy-registry

A curated, tested package registry for [guppylang](https://github.com/CQCL/guppylang)/HUGR: quantum algorithm building blocks, distributed as linkable [HUGR Packages](https://docs.quantinuum.com/guppy/api/defs.html).

## Why

HUGR (the compiler IR guppylang targets) already supports linking precompiled Packages into another program -- `guppylang.defs`'s `.emulator(libs: list[Package])` argument exists precisely for this. What's missing is an ecosystem of shared, genuinely-tested packages to link against: today, every guppylang project reimplements the same textbook building blocks (QFT, Grover, ansatze, ...) from scratch, at whatever quality bar that project's own tests demand.

This repo is a first step toward that ecosystem: a home for well-tested, documented guppylang implementations of standard algorithm components, each distributed both as importable guppy source (for projects that want to build on it in Python) and as a standalone compiled `.hugr` file (for projects that just want to link the compiled artifact via `libs=[...]`, no Python dependency on this repo at all).

## What's in it

- **[`packages/qft`](packages/qft/)** -- Quantum Fourier Transform and inverse-QFT. See its [README](packages/qft/README.md) for the API, and its [tests](packages/qft/tests/test_correctness.py) for how correctness is verified (against `numpy.fft`-derived reference matrices, run on Selene's statevector emulator).
- **[`packages/grover`](packages/grover/)** -- Grover's search algorithm (oracle-based amplitude amplification) on a 3-qubit, 1-marked-item register. See its [README](packages/grover/README.md) for the API, and its [tests](packages/grover/tests/test_correctness.py) for how correctness is verified (against the closed-form amplitude-amplification formula, run on Selene's statevector emulator).

More packages are planned -- see [CLAUDE.md](CLAUDE.md) for the current list and rationale.

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
    <future packages follow the same layout>
  README.md           this file
  CLAUDE.md            versions pinned, gotchas hit, testing methodology, roadmap
```

Each package under `packages/` is independent: its own `pyproject.toml`, its own tests, its own README. There's no shared root package -- `packages/qft`, `packages/grover` (and future packages) are meant to be installed and consumed individually.

## Install and use a package

Each package is a standard Python package with `guppylang` as a dependency. To use `qft` in your own project:

```
pip install -e path/to/guppy-registry/packages/qft
```

then, from your own guppy code:

```python
from qft import qft, iqft
```

Same pattern for `grover`: `pip install -e path/to/guppy-registry/packages/grover`, then `from grover import grover_search, oracle, diffuser`.

See [`packages/qft/README.md`](packages/qft/README.md) or [`packages/grover/README.md`](packages/grover/README.md) for the full API of each, and for how to link the precompiled `.hugr` package instead of depending on this repo's Python source.

## Development setup

Requires Python 3.12-3.14 (see [CLAUDE.md](CLAUDE.md) for the exact versions this repo was developed against).

```
python -m venv .venv
.venv/Scripts/activate   # or: source .venv/bin/activate on macOS/Linux
pip install -e "packages/qft[test]"
pip install -e "packages/grover[test]"
```

## Run the tests

```
pytest packages/qft/tests -v
pytest packages/grover/tests -v
```

Tests compile and run real guppy circuits against Selene's Quest (statevector) emulator and check the results against independently-computed reference values (see [CLAUDE.md](CLAUDE.md) for the full methodology, including why some `qft` tests deliberately run in subprocesses). Expect ~1-2 minutes per package -- each test case compiles and emulates a small quantum circuit.
