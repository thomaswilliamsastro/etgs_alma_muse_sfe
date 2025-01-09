import os

from sdss_psf import create_sdss_psf
from vars import wisdom_dir, galaxies, sdss_ref_dir, sdss_ref_im, sdss_psf_dir

os.chdir(wisdom_dir)

for galaxy in galaxies:
    if galaxy in sdss_ref_im:

        filename = os.path.join(sdss_ref_dir,
                                f"{galaxy}_r.fits"
                                )

        create_sdss_psf(filename,
                        psf_dir=sdss_psf_dir,
                        out_file=f"{galaxy}_r.psf.fits",
                        )

print('Complete!')
