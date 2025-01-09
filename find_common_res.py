import os

import numpy as np
from astropy.table import Table

from vars import wisdom_dir, galaxies, dists

os.chdir(wisdom_dir)

phys_reses = []
psf_tab = Table.read(os.path.join("muse_psf_match", "psf_tab.fits"))

print(np.nanmin(psf_tab["psf_arcsec"]),np.nanmax(psf_tab["psf_arcsec"]))

# for row in psf_tab:
#     print(f"{row['galaxy']}: {row['psf_arcsec']}")

for galaxy in galaxies:
    row = psf_tab[psf_tab["galaxy"] == galaxy][0]
    ang_res = row["psf_arcsec"]
    dist = dists[galaxy]

    phys_res = dist * 1e6 * np.radians(ang_res / 3600)
    phys_reses.append(phys_res)

phys_reses = np.array(phys_reses)
print(np.nanmax(phys_reses))

print("Complete!")
