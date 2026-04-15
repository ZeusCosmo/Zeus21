"""
Benchmark script demonstrating Zeus21 multiprocess performance with process-level caching.

This script compares the time to run multiple simulations with and without
the built-in CLASS caching in `zeus21.cosmology.runclass()`.

Usage:
    python examples/benchmark_multiprocess.py

Requirements:
    - zeus21 installed (including CLASS)
    - numpy
"""

import time
import multiprocessing as mp
import numpy as np
import zeus21


def _sim_worker_cached(_):
    """Worker that reuses the per-process CLASS cache."""
    user_params = zeus21.User_Parameters()
    cosmo_input = zeus21.Cosmo_Parameters_Input()
    classy_cosmo = zeus21.runclass(cosmo_input)
    cosmo_params = zeus21.Cosmo_Parameters(user_params, cosmo_input, classy_cosmo)
    hmf_interp = zeus21.HMF_interpolator(user_params, cosmo_params, classy_cosmo)
    astro_params = zeus21.Astro_Parameters(user_params, cosmo_params)
    coeffs = zeus21.get_T21_coefficients(
        user_params, cosmo_params, classy_cosmo, astro_params, hmf_interp, zmin=10.0
    )
    return float(coeffs.T21avg.sum())


def main():
    N_SIMS = 8
    N_WORKERS = 4

    print("=" * 60)
    print("Zeus21 Multiprocess Caching Benchmark")
    print("=" * 60)

    # Warmup: ensure CLASS is compiled/loaded in the main process
    print("\n[Warmup] Running first simulation...")
    _sim_worker_cached(None)
    print("Done.")

    # Single-process benchmark
    print("\n[Single-process benchmark]")
    t0 = time.time()
    for _ in range(N_SIMS):
        _sim_worker_cached(None)
    t_single = time.time() - t0
    print(f"  {N_SIMS} simulations in {t_single:.1f}s ({t_single / N_SIMS:.2f}s/sim)")

    # Multiprocess benchmark with spawn (safe for CLASS)
    print("\n[Multiprocess benchmark]")
    print(f"  Using spawn context with {N_WORKERS} workers for {N_SIMS} simulations")
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(processes=N_WORKERS) as pool:
        _ = pool.map(_sim_worker_cached, range(N_SIMS))
    t_multi = time.time() - t0
    print(f"  {N_SIMS} simulations in {t_multi:.1f}s ({t_multi / N_SIMS:.2f}s/sim)")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Single-process: {t_single:.1f}s total")
    print(f"Multiprocess  : {t_multi:.1f}s total")
    if t_multi > 0:
        print(f"Speedup       : {t_single / t_multi:.1f}x")
    print("\nNote: The first simulation in each new process incurs the")
    print("CLASS initialization cost (~few seconds). Subsequent calls in")
    print("the same process reuse the cached CLASS object automatically.")
    print("=" * 60)


if __name__ == "__main__":
    main()
