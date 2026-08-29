# vqe_h2

VQE (Variational Quantum Eigensolver) for the H2 molecule ground-state energy, implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package, with a real classical-quantum optimization loop. This package's correctness bar is higher than `qft`/`grover`/`qaoa`'s -- VQE/chemistry is Quantinuum's own flagship application area (InQuanto) -- see "How correctness was verified" below.

## The Hamiltonian

A 4-qubit, **Jordan-Wigner**-mapped qubit Hamiltonian for H2 in the STO-3G basis at (essentially) equilibrium bond length, taken verbatim (not re-derived) from PennyLane's official ["A brief overview of VQE"](https://pennylane.ai/demos/tutorial_vqe/) demo (Xanadu), which explicitly states it uses the Jordan-Wigner transformation. 15 Pauli terms; see `vqe_h2.py`'s module docstring for the full table and citation.

We independently verified this Hamiltonian before trusting it: exact diagonalization of the 16x16 matrix gives a ground-state energy of **-1.13726332 Hartree**, matching the widely-quoted literature H2 equilibrium ground-state energy of approximately -1.137 Hartree.

## What's here

- `ansatz_circuit(qs, params)` -- a hardware-efficient ansatz: Hartree-Fock reference state `|1100>` (X on qubits 0, 1), one layer of `RY(params[i])` per qubit, then a CNOT entangling ladder (0->1->2->3). 4 real parameters. Verified capable of reaching the *exact* ground state (not just "reasonably close") via a pure-numpy calibration performed before any guppy code was written -- see `vqe_h2.py`'s module docstring.
- `estimate_energy(params, shots=200, seed=0)` -- estimates `<psi(params)|H|psi(params)>` via real measurement statistics: 5 circuits (not 15 -- the 10 Z-only Hamiltonian terms are all estimated from one shared set of shots, a standard "qubit-wise commuting" grouping), each a real `ansatz_circuit` + basis-rotation + `measure_array`/`output`/`register_bitstrings()` cycle. Not a statevector shortcut.
- `optimize_vqe(shots=150, maxiter=20, x0=DEFAULT_X0)` -- the actual hybrid loop: `scipy.optimize.minimize` (Nelder-Mead) searching for the `ansatz_circuit` parameters that minimize `estimate_energy`. Returns `(best_params, best_energy)`.

## Install

From this directory:

```
pip install -e .
```

or add `vqe-h2 @ file:///path/to/guppy-registry/packages/vqe_h2` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from vqe_h2 import optimize_vqe, LITERATURE_GROUND_STATE_ENERGY

params, energy = optimize_vqe()
print(f"found {energy} Hartree, literature value is {LITERATURE_GROUND_STATE_ENERGY}")
```

## A hard-won lesson: the optimization landscape has many local minima

This ansatz's landscape has several local minima around -0.42 to -0.60 Hartree that both Nelder-Mead and BFGS get stuck in from "obvious" starting points (all-zero/Hartree-Fock, small perturbations of it, or several other arbitrary points we tried) -- checked directly against the exact, noiseless objective before writing any tests or picking a default. Of 5 independent random restarts, only 1 found the true minimum. `optimize_vqe`'s default `x0` (`DEFAULT_X0` in `vqe_h2.py`) is exactly that one validated, reproducible starting point -- not cherry-picked to flatter a result, it's the actual default every caller gets. If you change `x0`, be aware you may land in a local minimum instead; see `vqe_h2.py` and CLAUDE.md for the full investigation.

## Gotchas

- **A raw `numpy.ndarray` (what `scipy.optimize.minimize` always hands its objective) closed over and iterated inside a guppy body is rejected outright** -- convert it to a real Python `list` first. A `list` of `numpy.float64` elements is actually fine (verified precisely, see CLAUDE.md gotcha #19 for the full story, including an over-generalized first hypothesis we corrected before writing it up). `estimate_energy` converts every parameter with `float(p)` as cheap defense-in-depth regardless.
- The usual "no import inside a `@guppy` body", "can't reference `np.pi` inside a `@guppy` body" (precompute in Python, pass in via closure), etc. gotchas from `packages/qft`/`packages/grover`/`packages/qaoa` all apply here too -- nothing new on that front.

Full writeup, including which of `packages/qft`'s/`packages/grover`'s/`packages/qaoa`'s bug patterns were checked against this package (explicitly requested by the task brief) and found *not* to apply, is in the root [CLAUDE.md](../../CLAUDE.md).

## How correctness was verified

Given this package's higher rigor bar, every numeric claim is checked against an independent, from-scratch numpy reference (Pauli matrices + `np.kron`), not just the guppy code's own internal consistency:

1. The Hamiltonian's ground-state energy, via exact diagonalization, matches the literature value.
2. `ansatz_circuit`'s *exact* output (via `state_output`, no shot noise) matches a from-scratch numpy simulation of the same circuit, for several parameter sets.
3. The ansatz's expectation value `<psi|H|psi>`, computed from that exact statevector, matches exact diagonalization -- the task's explicit "verify against exact classical diagonalization for several parameter sets" requirement.
4. `estimate_energy` (the real, shot-sampled measurement pipeline) matches the same exact reference within shot-noise tolerance.
5. A live `optimize_vqe` run measurably beats the Hartree-Fock baseline.
6. A separate, larger-budget convergence check (using a cheaper single-circuit-per-evaluation exact-energy objective, not `estimate_energy`) confirms the ansatz can reach within a tight tolerance of the true ground state.

See `tests/test_correctness.py` for the full methodology.

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Expect this to take longer than `packages/qft`/`packages/grover`/`packages/qaoa`'s suites -- several tests run real optimization loops, not single circuit evaluations. Two tests in particular (`test_optimize_vqe_lowers_energy_from_hf_baseline`, `test_ansatz_can_reach_ground_state_given_enough_iterations`) are the highest-subprocess-spawn-volume tests in this entire registry; see their docstrings and CLAUDE.md gotcha #3 if they flake with `OSError: ... Application Control policy ...` (Windows only) -- that's an environmental issue unrelated to correctness.

## Examples

- `examples/basic_vqe.py` -- runs the classical-quantum loop and prints the found energy next to the Hartree-Fock baseline and the literature ground-state value.

Run with `python examples/basic_vqe.py` from this directory. Takes a couple of minutes (real optimization loop).
