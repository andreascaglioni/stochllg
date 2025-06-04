"""Plain tangent plane scheme (TPS) simulation. 
Now run on a square mesh with same problem as in paper"""


from math import sqrt
import os
import sys
from datetime import datetime
import shutil
from os.path import join
import numpy as np
from mpi4py import MPI

sys.path.insert(0, "./")  # Import from this project
from stochllg.BDF_FEM_TPS import BDF_FEM_TPS
from stochllg.utils import mesh_elems_area as mea
from stochllg.utils import export_xdmf, float_f


if __name__ == "__main__":
    # SETTINGS
    np.set_printoptions(formatter={"float_kind": float_f})
    comm = MPI.COMM_WORLD
    np.random.seed(1994)
    
    # PARAMETERS & DATA
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dir_save = join("simulations", "TPS_run_square_" + date + "/")
    print("Saving results in:", dir_save)
    os.makedirs(dir_save)
    shutil.copy(__file__, join(dir_save, "script.txt"))
    shutil.copy(join("data", "data_single_run_square.py"), join(dir_save, "data.txt"))
    from data.data_single_run_square import data  # noqa: E402

    n_MC_samples = 1
    dim_y = 1
    MC_samples = np.random.normal(0, 1, (n_MC_samples, dim_y))
    tt = data["tt"]
    dt = np.amax(tt[1:] - tt[:-1])
    msh = data["msh"]
    h = sqrt(np.amin(mea(msh)))

    # COMPUTE
    WW = data["W_fun"](MC_samples, tt)
    data["W"] = WW[0]
    print("Max dt:", float_f(dt))
    print("Min mesh size h:", float_f(h))
    
    mm, vv, ll, is_tt_ref = BDF_FEM_TPS(data, verbose=int(tt.size / 10))
    
    np.savetxt(join(dir_save, "MC_sample.csv"), MC_samples, delimiter=",")
    export_xdmf(msh, mm, tt, join(dir_save, "m.xdmf"))
    export_xdmf(msh, vv, tt, join(dir_save, "v.xdmf"))
    export_xdmf(msh, ll, tt, join(dir_save, "l.xdmf"))
