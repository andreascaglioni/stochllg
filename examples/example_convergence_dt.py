"""Convergence of TPS with respect to the time step size."""

from math import sqrt
import os
import sys
from datetime import datetime
import shutil
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.io import XDMFFile

sys.path.insert(0, "./")  # Import from this project
from src.BDF_FEM_TPS import BDF_FEM_TPS
from src.compute_mesh_elems_area import mesh_elems_area as mea
from src.utils import (
    error_space_time,
    export_xdmf,
    compute_rate,
    float_f,
)

if __name__ == "__main__":
    # SETTINGS
    np.set_printoptions(formatter={"float_kind": float_f})
    comm = MPI.COMM_SELF
    np.random.seed(0)

    # PARAMETERS & DATA
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dir_save = join("simulations", "BDF_FEM_conv_dt_" + date + "/")
    os.makedirs(dir_save)
    print("Saving results in:", dir_save)
    shutil.copy(__file__, join(dir_save, "script.txt"))
    from data.data_conv_dt import data  # noqa: E402

    shutil.copy(join("data", "data_conv_dt.py"), join(dir_save, "data.txt"))

    n_MC = 1
    dim_y = 1
    MC_sample = np.random.randn(dim_y)
    np.savetxt(join(dir_save, "MC_sample.csv"), MC_sample, delimiter=",")

    ddt = 0.01 * np.power(2.0, -np.arange(1, 8))  # array of time step sizes
    msh = data["msh"]
    h = sqrt(np.min(mea(msh)))
    ip_V3 = data["ip_V3"]

    # COMPUTE
    print("Sample reference solution")
    dt_ref = ddt[-1]  # reference time step size
    ddt = ddt[:-1]  # remove reference time step size
    T = data["T"]
    tt_ref = np.linspace(0, T, int(T / dt_ref) + 1)
    W_ref = data["W_fun"](tt_ref, MC_sample)
    # Add reference data to dictionary (make a deep copy)
    data["W"] = W_ref
    data["tt"] = tt_ref
    data["dtdt"] = tt_ref[1:] - tt_ref[:-1]
    print("dt:", float_f(dt_ref), "h:", float_f(h))
    mm_ref, _, _, _ = BDF_FEM_TPS(data, verbose=int(tt_ref.size / 5))
    xdmf = XDMFFile(comm, join(dir_save, "ref.xdmf"), "w")
    print("")

    print("Convergence Test:")
    err_tx = np.zeros_like(ddt)
    for i, dt in enumerate(ddt):
        print("dt:", float_f(dt), "h:", float_f(h))

        # Extract and store data in new data dict
        tt = np.linspace(0, T, int(T / dt) + 1)
        data["W"] = data["W_fun"](tt, MC_sample)
        data["tt"] = tt
        dtdt = tt[1:] - tt[:-1]

        # Compute
        mm, _, _, is_tt = BDF_FEM_TPS(data)

        # Error wrt reference
        err_tx[i], err_tt = error_space_time(
            mm_ref,
            tt_ref,
            mm,
            tt,
            ip_V3,
            t_error_type="Linf",
        )
        print("L^inf(0, T, H^1(D)) error:", float_f(err_tx[i]))
        print("")

        # Export sequence of time errors
        np.savetxt(join(dir_save, f"error_tt_{i}.csv"), err_tt, delimiter=",")
        export_xdmf(msh, mm, tt, join(dir_save, "m_" + str(i) + ".xdmf"))

        # Plot seqauecne of time errors
        plt.figure("error_t")
        plt.semilogy(tt_ref, err_tt, "-", label="dt = " + float_f(dt))

    # POST-PROCESS
    # print
    print("h (fixed): ", h)
    print("dt:", ddt, "reference: ", dt_ref)
    print("L^inf(0, T, H^1(D)) Error:", err_tx)
    rate = compute_rate(ddt, err_tx)
    print("Convergence rate:", rate)

    # Export data convergence
    A = np.vstack((h * np.ones_like(ddt), ddt, err_tx)).T
    np.savetxt(join(dir_save, "conv_data.csv"), A, delimiter=",", header="h, dt, error")

    # Plot
    plt.figure("error")
    plt.title("L^inf(0, T, H^1(D)) Error")
    plt.loglog(ddt, err_tx, ".-", label="error")
    C = err_tx[0] / (ddt[0])
    plt.loglog(ddt, C * ddt, "k-", label="C*dt")
    plt.legend()
    plt.xlabel("dt")
    plt.savefig(join(dir_save, "conv_error.png"))

    plt.figure("error_t")
    plt.xlabel("t")
    plt.title("H^1(D) error over time steps")
    plt.legend()
    plt.savefig(join(dir_save, "error_t.png"))

    plt.show()
