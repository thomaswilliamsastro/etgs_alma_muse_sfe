import glob
import os

import numpy as np
from mpdaf.obj import Cube
from pymusepipe.mpdaf_pipe import MuseCube

livefold = "/data/beegfs/astro-storage/groups/schinnerer/williams/muse/ETG"

target_sample = [
    "NGC0524",
    "NGC3489",
    "NGC3599",
    "NGC3607",
    "NGC3626",
    "NGC4435",
    "NGC4457",
    "NGC4596",
    "NGC4694",
    "NGC4697",
    "NGC7743",
]

target_res = "150pc"

# Distances in Mpc
dists = {
    'ngc0524': 23.3,
    "ngc3489": 11.86,
    "ngc3599": 19.86,
    'ngc3607': 22.2,
    "ngc3626": 20.05,
    'ngc4429': 16.5,
    'ngc4435': 16.7,
    'ngc4457': 15.10,
    "ngc4596": 15.76,
    "ngc4694": 15.76,
    'ngc4697': 11.4,
    "ngc7743": 20.32,
}

for target in target_sample:

    dist = dists[target.lower()]

    filename = glob.glob(os.path.join(livefold,
                                      "ETG",
                                      target,
                                      "Combined",
                                      "Cubes",
                                      "*DATACUBE_FINAL*WCS_Pall_mad*.fits"),
                         )
    if len(filename) > 1:
        raise IOError("Multiple copt files found. One of these must be wonky!")

    filename = filename[0]

    copt_res = float(os.path.split(filename)[-1].split("_")[-1].split("asec")[0])

    psf_array = ["gaussian",
                 copt_res,
                 2.8,
                 0,
                 6175,
                 ]

    mpcube = Cube(filename)
    cube = MuseCube(mpcube,
                    psf_array=psf_array)

    # Figure out the FWHM for the target physical resolution
    target_fwhm = float(target_res.split("pc")[0]) / (dist * 1e6)
    target_fwhm = np.degrees(target_fwhm) * 3600

    if target_fwhm <= copt_res:
        raise ValueError("target_fwhm is smaller than copt_res! This is a problem!")

    out_folder = os.path.join(livefold, "fixed_res", target)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_name = f"{target}_{target_res}.fits"

    cube.convolve_cube_to_psf(target_fwhm=target_fwhm,
                              outcube_folder=out_folder,
                              outcube_name=out_name,
                              )

    # Rebin by some integer factor to minimise space and remove so many unnecessary pixels
    rebin_factor = int(np.floor(target_fwhm / 0.2))
    out_name_rebin = out_name.replace(".fits", "_rebin.fits")
    out_name_rebin = os.path.join(out_folder, out_name_rebin)
    dap_name = out_name_rebin.replace(f"_{target_res}_rebin", "")

    mpcube = Cube(os.path.join(out_folder, out_name))
    cube = MuseCube(mpcube,
                    )

    cube.rebin_spatial(factor=rebin_factor, inplace=True)

    cube.write(out_name_rebin)

    print("Hardlinking so DAP knows the filename")
    os.system(f"ln {out_name_rebin} {dap_name}")

print("Complete!")
