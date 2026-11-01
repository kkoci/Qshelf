# qft

Quantum Fourier Transform and inverse-QFT, implemented in [guppylang](https://github.com/CQCL/guppylang) and distributed as a linkable HUGR Package.

## What's here

- `swap(q0, q1)` -- 3-CNOT qubit swap, daggerable.
- `cphase(control, target, theta)` -- controlled-phase gate CP(θ) = diag(1, 1, 1, e^{iθ}) (up to global phase), built as `rz(control, θ/2)` + `crz(control, target, θ)`. Daggerable. Note guppylang's `crz` alone is a controlled-Rz, *not* a controlled-phase gate.
- `qft[n](qs: array[qubit, n])` -- in-place QFT on an `n`-qubit register. `qs[0]` is the most significant qubit. Implements `QFT|j> = (1/sqrt(2**n)) * sum_k exp(2*pi*i*j*k/2**n) |k>`, i.e. exactly `numpy.fft.ifft(np.eye(2**n), norm="ortho")` in this ordering.
- `iqft[n](qs: array[qubit, n])` -- in-place inverse QFT, the exact adjoint of `qft`.

## Install

From this directory:

```
pip install -e .
```

or add `qft @ file:///path/to/packages/qft` to another project's dependencies. See the root [README](../../README.md) for the full dev environment setup.

## Use

```python
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit, discard_array
from qft import qft, iqft

@guppy
def my_circuit() -> None:
    qs = array(qubit() for _ in range(3))
    qft(qs)
    # ... do something in the Fourier basis ...
    iqft(qs)
    discard_array(qs)
```

`qft`/`iqft` are ordinary guppy functions -- call them directly from your own `@guppy` code as above, or link the compiled `.hugr` package into a separate program via `libs=[...]` (see `examples/build_lib.py` + `examples/linked_consumer.py`).

Note: guppylang#2250 closed, root cause was crz vs cphase in our code. Earlier versions of this package used `crz` where a controlled-phase gate was needed, which made `iqft` (and, invisibly to the old tests, `qft`) wrong by relative phases; `qft` and `iqft` can now each be used on their own.

## Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

Tests run real circuits on Selene's Quest (statevector) emulator and check them against `numpy.fft`-derived references, a hand-verified 2-qubit case, and round-trip identity checks -- both column-by-column on basis states and, more importantly, as full complex statevectors (up to one global phase) on random superposition inputs, which is what catches relative-phase errors. See `tests/test_correctness.py` for the full methodology.

## Examples

- `examples/basic_qft.py` -- QFT on a 3-qubit basis state, actual vs expected.
- `examples/roundtrip.py` -- QFT then inverse-QFT returns the original state.
- `examples/build_lib.py` + `examples/linked_consumer.py` -- compile `qft` to a standalone `.hugr` file and link it into a separate consumer program via `libs=[...]`, with no Python import of this package.

Run any of them with `python examples/<name>.py` from this directory (`build_lib.py` first, if running `linked_consumer.py`).
