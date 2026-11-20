# TTG_DoS, branch `julia`

This branch holds only the Julia package [`UnrelaxedTTG.jl`](UnrelaxedTTG.jl/): the momentum-space
tight-binding model of unrelaxed, doubly incommensurate twisted trilayer graphene, the W/L truncation
and the kernel polynomial method for the momentum LDoS and the density of states, on the CPU and on
an NVIDIA GPU. See [`UnrelaxedTTG.jl/README.md`](UnrelaxedTTG.jl/README.md) for installation, use
and the drivers in `examples/`.

The MATLAB code (the earlier implementation, with `testDoS.m`, `DoS/`, `Hamiltonians/`, `cluster/`
and the rest) is on the branch `main`, unchanged.
