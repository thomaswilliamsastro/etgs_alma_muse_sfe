import os

from mpdaf.obj import Cube

from vars import wisdom_dir, muse_r_band_dir, galaxies, muse_cube_dir

os.chdir(wisdom_dir)

if not os.path.exists(muse_r_band_dir):
    os.makedirs(muse_r_band_dir)

for galaxy in galaxies:
    cube_name = os.path.join(muse_cube_dir,
                             f"{galaxy.upper()}",
                             "Combined",
                             "Cubes",
                             f"{galaxy.upper()}_DATACUBE_FINAL_P001.fits"
                             )

    out_file = os.path.join(muse_r_band_dir,
                            f"{galaxy}_muse_r.fits"
                            )

    if not os.path.exists(out_file):
        cube = Cube(cube_name)
        r_im = cube.get_band_image("SDSS_r")
        r_im.write(out_file)

print("Complete!")
