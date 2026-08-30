"""VQE (Variational Quantum Eigensolver) for the H2 molecule ground-state
energy at a STRETCHED (non-equilibrium) bond length, in guppylang.

This package exists to answer one question, not to add a new molecule for
its own sake: `packages/vqe_h2` documented that its hardware-efficient
ansatz's optimization landscape has many local minima, and that an "obvious"
starting point (all-zero/Hartree-Fock, or small perturbations of it) gets
stuck. Is that an H2-*equilibrium*-specific quirk, or a general property of
this ansatz style? See "The local-minima finding, reproduced" below for the
answer (short version: general -- it reproduces here too, with a different,
also-empirically-characterized texture).

The Hamiltonian
----------------
A 4-qubit, Jordan-Wigner-mapped qubit Hamiltonian for H2 in the STO-3G
basis, same molecule/basis/mapping as `packages/vqe_h2`, but at bond length
**2.1 Angstrom** (~2.83x the ~0.742A equilibrium bond length `vqe_h2` uses)
-- a genuinely stretched, strongly-correlated geometry, well past the point
where a single Slater determinant (Hartree-Fock) is a good reference state.

Per the task brief, this was **not** derived by hand -- it's cited from the
same literature/framework-standard source as `packages/vqe_h2`'s
Hamiltonian: PennyLane's `qchem` dataset service
(https://pennylane.ai/datasets, Xanadu), specifically
`qml.data.load('qchem', molname="H2", basis="STO-3G", bondlength=2.1)[0].hamiltonian`.
We did **not** re-run `qml.data.load` "live" from inside this package (this
registry pins `guppylang`/`pytest`/`numpy`/`scipy` only, see root
CLAUDE.md's "Versions used") -- the 15-term table below was fetched once via
that exact call, in an isolated scratch environment, and is now cited
verbatim, the same relationship `vqe_h2.py` has to PennyLane's `tutorial_vqe`
demo page.

**Confirmed this is the same generation pipeline/mapping `vqe_h2` cites,
not just "also PennyLane, presumably fine": ** `qml.data.load('qchem',
molname="H2", basis="STO-3G", bondlength=0.742)[0].hamiltonian` -- the same
dataset *family*, just requesting the equilibrium bond length instead of a
stretched one -- reproduces `vqe_h2.HAMILTONIAN`'s 15 coefficients exactly,
digit for digit (e.g. the identity coefficient is -0.09963387941370971 in
both). Since `vqe_h2.py`'s module docstring already independently
established (and this package doesn't re-derive) that this dataset's H2
Hamiltonians are Jordan-Wigner-mapped, this exact match confirms the
STRETCHED table below comes from that identical, already-JW-verified
pipeline, just evaluated at a different geometry -- not a different mapping
or a different tool that happens to look similar.

We independently verified the stretched Hamiltonian before trusting it, the
same rigor `vqe_h2` applied: exact diagonalization of the 16x16 matrix below
gives a ground-state energy of -0.944374524987463 Hartree, matching
PennyLane's own `fci_energy` field for this exact dataset
(-0.9443745249874615 Hartree) to ~1e-15 -- see
`tests/test_correctness.py::test_hamiltonian_ground_state_matches_pennylane_fci`
and CLAUDE.md. As expected for a stretched geometry, this is a much shallower
binding energy than equilibrium H2's ~-1.137 Hartree (a separated H + H pair
in this basis approaches -1.0 Hartree; 2.1A is well on the way there).

Qubit convention: identical to `vqe_h2` -- qubit 0 is the leftmost/most-
significant tensor factor; Z_i in a term below means "apply Z to `qs[i]`".

Ansatz
------
Identical to `packages/vqe_h2`'s `ansatz_circuit`, unchanged: Hartree-Fock-
style reference state (X on qubits 0 and 1, `|1100>` in this ordering, same
occupied-orbital convention the Hamiltonian uses), one layer of
`RY(params[i])` on each qubit, then a CNOT ladder (0->1, 1->2, 2->3). Reusing
the exact same ansatz (not a redesigned one) is what makes this package's
central comparison meaningful -- any difference found is attributable to the
Hamiltonian/geometry, not a confound from also changing the circuit.

Estimating the energy
----------------------
Identical machinery and citation basis to `packages/vqe_h2`'s
`estimate_energy` -- same measurement-basis-rotation approach (`h(q)` for X,
`sdg(q); h(q)` for Y, nothing for Z/identity), same qubit-wise-commuting
grouping of this Hamiltonian's 10 Z-only terms into a single shared circuit,
same 5-circuits-per-evaluation shape (this Hamiltonian has the same term
*shape* as `vqe_h2`'s -- 1 identity + 4 single-Z + 6 ZZ + 4 double-excitation
X/Y terms -- just different coefficients, so the grouping logic transfers
unchanged). See `vqe_h2/vqe_h2.py`'s module docstring for the full
derivation of why this is correct; not re-derived here.

The local-minima finding, reproduced
--------------------------------------
`vqe_h2.DEFAULT_X0` exists because, checked directly against the exact
(noiseless) objective, most "obvious" starting points for the equilibrium
H2 landscape get stuck around -0.42 to -0.60 Hartree instead of the true
-1.137 Hartree minimum; only 1 of 5 random restarts (BFGS, run to
convergence) found the true minimum.

We ran the exact same experiment here, against this stretched Hamiltonian,
before writing any of this package's tests -- see CLAUDE.md for the full
numbers. Findings:

- **All-zero start and small perturbations of it get stuck, exactly like
  `vqe_h2`.** BFGS and Nelder-Mead from `[0, 0, 0, 0]`, and from 5 small
  random perturbations of it (`N(0, 0.1)` per parameter), all converge to
  the *same* local minimum, -0.9269926 Hartree -- not the true -0.9443745
  Hartree minimum.
- **Random-restart recovery rate is of the same order as `vqe_h2`'s.** Of 10
  independent `uniform(-pi, pi, 4)` restarts (BFGS to convergence), only
  2 found the true minimum (`vqe_h2`: 1 of 5) -- confirming this isn't an
  H2-equilibrium-specific artifact: **the same ansatz style genuinely has a
  landscape with a dominant, hard-to-escape local minimum at a completely
  different geometry too.**
- **A real, quantitative difference from `vqe_h2`, also worth recording:**
  this landscape's trap is *shallower* than equilibrium H2's. `vqe_h2`'s
  local minima sit ~0.5-0.7 Hartree above its true minimum; here, the
  dominant trap (-0.9269926) is only ~0.017 Hartree above the true minimum
  (-0.9443745) -- a much smaller, easier-to-miss gap. Naively checking "did
  the optimizer land near a good-looking energy" would *not* have caught
  this landscape's local-minimum problem the way it obviously does for
  `vqe_h2`; only checking against the *known* true minimum (independently
  computed via exact diagonalization, per this registry's standing
  methodology) reveals it.
- **A second, related difference: even a validated good starting point
  needs more optimizer iterations here than `vqe_h2`'s did.** `vqe_h2`'s
  `DEFAULT_X0` reaches within ~0.02 Hartree after just 20 Nelder-Mead
  iterations. Every validated-good seed we found for *this* landscape (3 of
  30 random restarts reached the true minimum via Nelder-Mead run to 80
  iterations) was still >0.1 Hartree off at 20 iterations and needed
  60-80 to fully converge -- so `optimize_vqe`'s default `maxiter` is 60
  here, not `vqe_h2`'s 20 (see `DEFAULT_X0` below and CLAUDE.md for the full
  sweep).
- **A third difference, found while validating `optimize_vqe` against the
  REAL (shot-sampled) pipeline, not just the exact objective: this shallow
  landscape is noticeably more shot-noise-sensitive than `vqe_h2`'s.** A
  first attempt at this package's `optimize_vqe` smoke test used
  `shots=80`, `maxiter=25` (analogous to `vqe_h2`'s `shots=80, maxiter=8`)
  and failed a "clearly better than baseline" assertion twice in a row --
  not an environmental flake (checked -- no `WinError 4551`), a real,
  reproducible effect: at `shots=80`/`150`, per-evaluation shot noise on
  `estimate_energy` is large *relative to this landscape's shallow ~0.017
  Hartree local-minimum gap*, enough to substantially derail Nelder-Mead's
  real trajectory even though the *noiseless* objective at the same
  `maxiter` clearly descends (~-0.92 Hartree by iteration 25, vs this
  landscape's -0.51 baseline). Raising `shots` to 500 (cheap: more samples
  from the same simulated statevector per circuit, not more circuit
  compiles/launches) fixed it. `vqe_h2`'s much deeper local-minima gaps
  apparently make its optimization comparatively robust to shot noise at
  low shot counts; this package's shallower landscape is not -- a
  genuinely new, landscape-shape-dependent finding this package surfaced
  that `vqe_h2` had no reason to encounter. See CLAUDE.md for the full
  before/after numbers.

See CLAUDE.md's "vqe_h2_stretched package notes" section for the full
writeup, including the complete 30-trial sweep and why this landscape's
local-minimum texture (one dominant, moderately-deep basin) differs from
equilibrium H2's (several distinct, much-deeper basins) even though both
clearly need a validated starting point.

Checked against CLAUDE.md's full bug/finding list before writing any of the
above, per the task brief and this registry's standing convention -- see
CLAUDE.md for the full writeup (short version: nothing new: this package
reuses `vqe_h2`'s circuit structure unchanged, so every gotcha check
`vqe_h2` already did -- #5, #12, #13, #16, #17 not applicable; #18 not
applicable to `ansatz_circuit` itself, real-valued; #19 applies and is
handled the same way -- transfers over unchanged).
"""

import math
import time

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    h,
    measure_array,
    qubit,
    ry,
    sdg,
    x,
)

N_QUBITS = 4
_PI = math.pi

# The stretched-H2/STO-3G Jordan-Wigner Hamiltonian at bond length 2.1A,
# fetched verbatim (see module docstring for citation and independent
# verification) via
# `qml.data.load('qchem', molname="H2", basis="STO-3G", bondlength=2.1)[0].hamiltonian`.
# Same {qubit_index: 'X'|'Y'|'Z'} term-dict convention as `vqe_h2.HAMILTONIAN`.
HAMILTONIAN: list[tuple[float, dict[int, str]]] = [
    (-0.5371948795513937, {}),
    (0.06358459365791537, {0: "Z"}),
    (0.06358459365791537, {1: "Z"}),
    (0.12588461015922925, {0: "Z", 1: "Z"}),
    (0.06607344554223991, {0: "Y", 1: "X", 2: "X", 3: "Y"}),
    (-0.06607344554223991, {0: "Y", 1: "Y", 2: "X", 3: "X"}),
    (-0.06607344554223991, {0: "X", 1: "X", 2: "Y", 3: "Y"}),
    (0.06607344554223991, {0: "X", 1: "Y", 2: "Y", 3: "X"}),
    (0.011724879133723665, {2: "Z"}),
    (0.06219166527579956, {0: "Z", 2: "Z"}),
    (0.011724879133723665, {3: "Z"}),
    (0.12826511081803946, {0: "Z", 3: "Z"}),
    (0.12826511081803946, {1: "Z", 2: "Z"}),
    (0.06219166527579956, {1: "Z", 3: "Z"}),
    (0.13176640290211938, {2: "Z", 3: "Z"}),
]

#: Bond length this Hamiltonian was generated at, Angstrom -- for reference
#: only (not used anywhere in the circuit; the Hamiltonian already bakes it
#: in). Equilibrium is ~0.742A (see `vqe_h2.py`); this is ~2.83x that.
BOND_LENGTH_ANGSTROM = 2.1

# Exact ground-state energy for this Hamiltonian, Hartree. This is
# PennyLane's own `fci_energy` field for this exact dataset, independently
# confirmed by exact diagonalization of HAMILTONIAN above (see module
# docstring and tests): np.linalg.eigvalsh gives -0.944374524987463,
# matching to ~1e-15.
EXACT_GROUND_STATE_ENERGY = -0.9443745249874615


@guppy
def ansatz_circuit(qs: array[qubit, N_QUBITS], params: array[angle, N_QUBITS]) -> None:
    """Hardware-efficient ansatz, unchanged from `packages/vqe_h2`: Hartree-
    Fock-style reference |1100>, one layer of RY(params[i]) per qubit, then a
    CNOT entangling ladder 0->1->2->3."""
    x(qs[0])
    x(qs[1])
    for i in range(N_QUBITS):
        ry(qs[i], params[i])
    for i in range(N_QUBITS - 1):
        cx(qs[i], qs[i + 1])


def _z_group_terms() -> list[tuple[float, dict[int, str]]]:
    return [(c, t) for c, t in HAMILTONIAN if t and all(p == "Z" for p in t.values())]


def _rotation_terms() -> list[tuple[float, dict[int, str]]]:
    return [(c, t) for c, t in HAMILTONIAN if any(p != "Z" for p in t.values())]


def _build_measurement_circuit(term: dict[int, str], params: list[float]):
    """Build a guppy entrypoint: `ansatz_circuit(params)`, then basis
    rotations for `term`'s Pauli type on each qubit it touches, then measure
    everything. Identical structure to `vqe_h2._build_measurement_circuit`."""

    @guppy.comptime
    def main() -> None:
        qs = array(qubit() for _ in range(N_QUBITS))
        param_angles = array(angle(p / _PI) for p in params)
        ansatz_circuit(qs, param_angles)
        for i in range(N_QUBITS):
            pauli = term.get(i)
            if pauli == "X":
                h(qs[i])
            elif pauli == "Y":
                sdg(qs[i])
                h(qs[i])
        bits = collect_measurements(measure_array(qs))
        output("bits", bits)

    return main


def _term_eigenvalue(bitstring: str, term: dict[int, str]) -> int:
    value = 1
    for i in term:
        if bitstring[i] == "1":
            value = -value
    return value


def _run_circuit_and_get_bitstrings(term: dict[int, str], params: list[float], shots: int, seed: int) -> list[str]:
    main = _build_measurement_circuit(term, params)
    main.check()
    emulator = main.with_minimal_opt().emulator(n_qubits=N_QUBITS).statevector_sim().with_seed(seed).with_shots(shots)
    result = None
    last_error: OSError | None = None
    for attempt in range(10):
        try:
            result = emulator.run()
            break
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0 * (attempt + 1), 6.0))
    if result is None:
        raise RuntimeError(f"emulator run failed after retries: {last_error}") from last_error
    return result.register_bitstrings()["bits"]


def estimate_energy(params: list[float], shots: int = 200, seed: int = 0) -> float:
    """Estimate <psi(params)|H|psi(params)> via real measurement statistics,
    identical pipeline to `vqe_h2.estimate_energy` (5 circuits: 1 shared for
    the Z-only term group, 1 each for the 4 double-excitation terms)."""
    # See CLAUDE.md gotcha #19: convert to a real Python list of plain
    # floats before closing over it inside a guppy body -- a raw
    # numpy.ndarray (what scipy.optimize.minimize always hands its
    # objective) is rejected outright.
    params = [float(p) for p in params]
    energy = sum(c for c, t in HAMILTONIAN if not t)  # identity term

    z_terms = _z_group_terms()
    if z_terms:
        bitstrings = _run_circuit_and_get_bitstrings({}, params, shots, seed)
        for coeff, term in z_terms:
            avg = sum(_term_eigenvalue(b, term) for b in bitstrings) / len(bitstrings)
            energy += coeff * avg

    for coeff, term in _rotation_terms():
        bitstrings = _run_circuit_and_get_bitstrings(term, params, shots, seed)
        avg = sum(_term_eigenvalue(b, term) for b in bitstrings) / len(bitstrings)
        energy += coeff * avg

    return energy


#: A validated, reproducible default starting point for `optimize_vqe`
#: (`np.random.default_rng(13).uniform(-pi, pi, 4)`). Same methodology as
#: `vqe_h2.DEFAULT_X0`: checked directly against the exact, noiseless
#: objective before picking a default. This landscape also has a real,
#: hard-to-escape local minimum (-0.9269926 Hartree, vs the true -0.9443745
#: Hartree) -- the all-zero start, small perturbations of it, and most
#: (25/30 in a wider sweep) random restarts land there instead. This
#: specific seed is one of only 3/30 that reached the true minimum via
#: Nelder-Mead run to convergence -- but even it needs ~60-80 iterations to
#: get there (this landscape converges more slowly than vqe_h2's from a
#: validated seed, not just harder to find one), hence `optimize_vqe`'s
#: default `maxiter=60` below, not vqe_h2's 20. See CLAUDE.md and
#: tests/test_correctness.py for the full investigation.
DEFAULT_X0 = [2.292090838837183, 2.2324315414250684, 1.954217649430836, -1.4988767169222235]


def optimize_vqe(
    shots: int = 500,
    seed: int = 0,
    maxiter: int = 60,
    x0: list[float] | None = None,
) -> tuple[list[float], float]:
    """Classical optimization loop (plain Python, outside guppy), identical
    in structure to `vqe_h2.optimize_vqe` -- `scipy.optimize.minimize`
    (Nelder-Mead) searching for the ansatz parameters minimizing the
    estimated stretched-H2 energy. Two defaults differ from `vqe_h2`'s, both
    because this landscape's dominant local-minimum trap is *shallow*
    (~0.017 Hartree deep, vs. vqe_h2's ~0.5-0.7 Hartree-deep traps -- see
    module docstring): `maxiter=60` not 20 (needs more iterations even from
    a validated seed), and `shots=500` not `vqe_h2`'s 150 (discovered
    empirically, not assumed: at 80-150 shots, per-evaluation shot noise on
    `estimate_energy` is large enough relative to this shallow landscape's
    signal that Nelder-Mead's real, noisy trajectory diverges substantially
    from what the noiseless objective predicts -- an early version of this
    package's smoke test used shots=80/maxiter=25 and failed to beat the
    baseline by a comfortable margin twice in a row, even though the
    noiseless objective at the same maxiter reaches ~-0.92 Hartree; raising
    shots to 500 (cheap -- more samples from the same simulated statevector,
    not more circuit compiles) closed the gap. See CLAUDE.md for the full
    investigation.

    Returns (best_params, best_energy).
    """
    from scipy.optimize import minimize

    if x0 is None:
        x0 = DEFAULT_X0

    def objective(x: list[float]) -> float:
        return estimate_energy(list(x), shots=shots, seed=seed)

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 0.05, "fatol": 0.001},
    )
    best_params = list(result.x)
    best_energy = estimate_energy(best_params, shots=shots, seed=seed)
    return best_params, best_energy
