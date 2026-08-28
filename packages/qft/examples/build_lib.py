"""Compile `qft` (monomorphized for a 3-qubit register) to a standalone,
distributable HUGR Package file: qft3.hugr.

This is what it means to "distribute" a guppy-registry package: a `.hugr` file
that any other guppylang program can link against with
`<entrypoint>.emulator(libs=[...])`, without needing this package's Python
source at all (see linked_consumer.py). Run this first, then
linked_consumer.py, from the packages/qft directory.
"""

from guppylang import guppy
from guppylang.library import link_name
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit

from qft import qft

N = 3


@guppy
@link_name("qft_registry.qft3")
def qft3(qs: array[qubit, N]) -> None:
    """Concrete (monomorphized) 3-qubit QFT, exported under a stable link name."""
    qft(qs)


def build() -> None:
    package = qft3.with_minimal_opt().compile_function()
    # Mark the package as a library (no fixed entrypoint) so it can be linked
    # into another package's emulator build. See ../../../CLAUDE.md.
    for module in package.modules:
        module.entrypoint = module.module_root
    with open("qft3.hugr", "wb") as f:
        f.write(package.to_bytes())
    print(f"wrote qft3.hugr ({len(package.modules)} module(s))")


if __name__ == "__main__":
    build()
