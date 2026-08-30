# vqe_h2_stretched

VQE (Variational Quantum Eigensolver) for the H2 molecule ground-state energy at a **stretched, non-equilibrium bond length** (2.1 Angstrom, vs. `packages/vqe_h2`'s ~0.742A equilibrium), implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package. Same chemistry rigor bar as `vqe_h2` -- see "How correctness was verified" below.

## Why this package exists

`packages/vqe_h2` found that its hardware-efficient ansatz's optimization landscape has many local minima -- an "obvious" starting point (all-zero/Hartree-Fock, or small perturbations of it) reliably gets stuck. That finding was made on exactly one Hamiltonian: H2 at equilibrium. **Is that landscape shape an H2-equilibrium-specific quirk, or a general property of this ansatz style?** This package answers that by rerunning the exact same investigation against a genuinely different Hamiltonian -- same molecule, same ansatz, same code structure, only the geometry (and therefore the Hamiltonian) changes.

**Packaging choice, documented as the task brief asked: a new, independent `packages/vqe_h2_stretched`, not a "second-molecule mode" bolted onto `packages/vqe_h2`.** Consistent with every other second-round addition in this registry (`vqe_h2` didn't touch `qaoa`, `repetition_code` didn't touch `grover`, `grover_multi` didn't touch `grover`), a new package keeps `vqe_h2`'s already-published, tested API untouched. It also keeps this package's own central claim self-contained and falsifiable on its own terms: if `vqe_h2_stretched` reused `vqe_h2`'s module and just swapped a Hamiltonian constant behind a flag, a bug in the shared plumbing could quietly affect both packages' conclusions together. As independent packages, each package's tests only need to trust its own code.

**Why "H2 at a stretched bond length" instead of LiH** (the task brief's other suggested option): LiH's qubit Hamiltonian, even with an active-space reduction, needs 6-10 qubits and a citable, framework-standard literal Hamiltonian table proved hard to source with the same rigor `vqe_h2`'s citation had (checked -- no PennyLane demo we found prints one; see CLAUDE.md). H2 at a different bond length keeps the exact same 4-qubit ansatz and Hamiltonian *shape* as `vqe_h2` (only the 15 coefficients differ), which is also methodologically cleaner for this package's actual question: reusing the identical circuit means any difference found is attributable to the landscape/Hamiltonian, not a confound from also changing the ansatz's qubit count or structure.

## The Hamiltonian

A 4-qubit, **Jordan-Wigner**-mapped qubit Hamiltonian for H2 in the STO-3G basis at bond length **2.1A** (~2.83x `vqe_h2`'s ~0.742A equilibrium length) -- a genuinely stretched, strongly-correlated geometry. Cited, not derived, from the same framework-standard source `vqe_h2` used: PennyLane's [`qchem` dataset service](https://pennylane.ai/datasets), specifically `qml.data.load('qchem', molname="H2", basis="STO-3G", bondlength=2.1)[0].hamiltonian`. See `vqe_h2_stretched.py`'s module docstring for the full citation and, importantly, **independent confirmation this is the same generation pipeline/mapping `vqe_h2` cites**: requesting the identical dataset service at the equilibrium bond length (0.742A) reproduces `vqe_h2.HAMILTONIAN`'s 15 coefficients exactly, digit for digit.

We independently verified this Hamiltonian before trusting it, same as `vqe_h2`: exact diagonalization of the 16x16 matrix gives a ground-state energy of **-0.944374524987463 Hartree**, matching PennyLane's own `fci_energy` field for this dataset to ~1e-15.

## What's here

Identical API shape to `vqe_h2` (see that package's README for the full description of each function -- not repeated here):

- `ansatz_circuit(qs, params)` -- unchanged from `vqe_h2`.
- `estimate_energy(params, shots=200, seed=0)` -- unchanged pipeline; different `HAMILTONIAN`.
- `optimize_vqe(shots=500, maxiter=60, x0=DEFAULT_X0)` -- **two defaults differ from `vqe_h2`'s** (`shots=150`, `maxiter=20`); see "The finding" below for why.

## Install

From this directory:

```
pip install -e .
```

or add `vqe-h2-stretched @ file:///path/to/guppy-registry/packages/vqe_h2_stretched` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from vqe_h2_stretched import optimize_vqe, EXACT_GROUND_STATE_ENERGY

params, energy = optimize_vqe()
print(f"found {energy} Hartree, exact ground state is {EXACT_GROUND_STATE_ENERGY}")
```

## The finding: local minima are NOT an H2-equilibrium-specific quirk

Reran `vqe_h2`'s exact investigation -- starting from all-zero parameters, small perturbations of it, and random restarts, checked directly against the exact (noiseless) objective -- against this stretched Hamiltonian:

- **All-zero and small perturbations get stuck, exactly like `vqe_h2`.** Both land at the same local minimum, -0.9269926 Hartree, not the true -0.9443745 Hartree minimum.
- **Random-restart recovery rate is the same order of magnitude as `vqe_h2`'s.** Of 10 independent `uniform(-pi, pi, 4)` restarts (BFGS), only 2 found the true minimum (`vqe_h2`: 1 of 5). **This directly answers the task's question: the same ansatz style genuinely produces a landscape with a dominant, hard-to-escape local minimum at a completely different molecular geometry too -- it generalizes.**
- **But the texture is quantitatively different, not identical.** This landscape's trap is much *shallower* than `vqe_h2`'s: only ~0.017 Hartree above the true minimum (vs. `vqe_h2`'s local minima, ~0.5-0.7 Hartree above its true minimum). A wider 30-trial sweep against Nelder-Mead (the method `optimize_vqe` actually uses) found only 3/30 reaching the true minimum, and even those needed ~60-80 iterations to fully converge -- `vqe_h2`'s validated seed reaches within ~0.02 Hartree after just 20.
- **A second, newly-discovered difference, found while validating the REAL (shot-sampled) pipeline: this landscape is noticeably more shot-noise-sensitive.** Because the local-minimum trap is so shallow, the "correct" downhill signal is easily swamped by shot noise at low shot counts -- an early version of this package's `optimize_vqe` smoke test, using `vqe_h2`-equivalent settings (`shots=80`), failed a "clearly improves on baseline" check twice in a row, a real and reproducible effect (not the usual environmental `WinError 4551` flake -- checked). Raising `shots` to 500 fixed it; `optimize_vqe`'s default reflects that.

`DEFAULT_X0` in `vqe_h2_stretched.py` is a validated, reproducible starting point found the same way `vqe_h2.DEFAULT_X0` was (checked against the exact objective before being adopted as the default every caller gets). See `vqe_h2_stretched.py`'s module docstring and the root [CLAUDE.md](../../CLAUDE.md) for the full investigation and numbers.

## Gotchas

Nothing new -- this package reuses `vqe_h2`'s circuit structure unchanged, so every guppylang-specific finding `vqe_h2` already checked (gotchas #5, #12, #13, #16, #17 not applicable; #18 not applicable to `ansatz_circuit` itself; #19 applies, handled the same way) transfers over unmodified. See `vqe_h2/README.md`'s Gotchas section and the root [CLAUDE.md](../../CLAUDE.md) for the full writeup. This package's own contribution isn't a new guppylang bug -- it's the shot-noise-sensitivity and slower-convergence findings above, which are properties of this specific optimization landscape, not of guppylang.

## How correctness was verified

Same higher rigor bar as `vqe_h2` -- every numeric claim checked against an independent, from-scratch numpy reference, not just the guppy code's own internal consistency:

1. The Hamiltonian's ground-state energy, via exact diagonalization, matches PennyLane's own `fci_energy` field for this dataset.
2. The cited Hamiltonian is confirmed to be the same PennyLane dataset family/generation pipeline `vqe_h2` used (byte-for-byte match at the equilibrium bond length).
3. `ansatz_circuit`'s exact output (`state_output`, no shot noise) matches a from-scratch numpy simulation, for several parameter sets.
4. The ansatz's expectation value `<psi|H|psi>`, computed from the exact statevector, matches exact diagonalization.
5. `estimate_energy` (the real, shot-sampled measurement pipeline) matches the exact reference within shot-noise tolerance.
6. The local-minimum trap is directly demonstrated (all-zero/small-perturbation starts provably get stuck) and `DEFAULT_X0` is directly validated to escape it, both against the exact objective -- the actual point of this package.
7. A live `optimize_vqe` run measurably beats the params=0 baseline.
8. A separate, larger-budget convergence check confirms the ansatz can reach the true ground state given enough iterations.

See `tests/test_correctness.py` for the full methodology.

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Expect this to take longer than most packages in this registry -- several tests run real optimization loops. `test_optimize_vqe_lowers_energy_from_baseline` and `test_ansatz_can_reach_ground_state_given_enough_iterations` are the heaviest (real live optimizer runs); see CLAUDE.md gotcha #3 if they flake with `OSError: ... Application Control policy ...` (Windows only, environmental, unrelated to correctness).

## Examples

- `examples/basic_vqe_stretched.py` -- runs the classical-quantum loop and prints the found energy next to the params=0 baseline and the exact ground-state value.

Run with `python examples/basic_vqe_stretched.py` from this directory. Takes several minutes (real optimization loop, `maxiter=60` by default -- see "The finding" above for why).
