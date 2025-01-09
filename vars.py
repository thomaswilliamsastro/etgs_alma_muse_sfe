import copy
import os
import socket

import seaborn as sns
from uncertainties import ufloat
from astropy.table import Table

host = socket.gethostname()

if "mac" in host:
    wisdom_dir = "/Users/thomaswilliams/Documents/wisdom"
elif "astro" in host:
    wisdom_dir = "/data/beegfs/astro-storage/groups/schinnerer/williams/wisdom"
else:
    raise Warning(f"host {host} not recognised")

matched_res = "150pc"

# PGCS to NGCs for z0MGS
pgc_mapping = {
    "ngc0524": 5222,
    "ngc3489": 33160,
    "ngc3599": 34326,
    "ngc3607": 34426,
    "ngc3626": 34684,
    "ngc4429": 40850,
    "ngc4435": 40898,
    "ngc4457": 41101,
    "ngc4596": 42401,
    "ngc4694": 43241,
    "ngc4697": 43276,
    "ngc7743": 72263,
}

z0mgs_tab = Table.read("/Users/thomaswilliams/Documents/wisdom/z0mgs_dr1_index.fits")

# Pull out integrated SFRs
int_sfrs = {}
for ngc in pgc_mapping:
    row = z0mgs_tab[z0mgs_tab["PGC"] == pgc_mapping[ngc]][0]
    sfr_val = row["LOGSFR"]
    err = row["E_LOGSFR"]

    sfr = ufloat(sfr_val, err)
    int_sfrs[ngc] = copy.deepcopy(sfr)

#     print(ngc, sfr)
# no

galaxies = [
    "ngc0524",
    # "ngc1317",
    "ngc3489",
    "ngc3599",
    "ngc3607",
    "ngc3626",
    "ngc4435",
    # "ngc4457",
    "ngc4596",
    "ngc4694",
    "ngc4697",
    "ngc7743",
]

alpha_co = 4.35 / 0.7

# Distances in Mpc
dists = {
    'ngc0524': 23.3,
    "ngc3489": 11.86,
    "ngc3599": 19.86,
    'ngc3607': 22.2,
    "ngc3626": 20.05,
    'ngc4429': 16.5,
    'ngc4435': 16.7,
    "ngc4596": 15.76,
    "ngc4694": 15.76,
    'ngc4697': 11.4,
    "ngc7743": 20.32,
}

# ALMA file mappings
alma_files = {"ngc0524": f"ngc0524_12m+7m_co21_2p5kms_{matched_res}_strict",
              "ngc3489": f"ngc3489_12m+7m+tp_co21_2p5kms_{matched_res}_strict",
              "ngc3599": f"ngc3599_12m+7m+tp_co21_2p5kms_{matched_res}_strict",
              "ngc3607": f"ngc3607_12m+7m_co21_2p5kms_{matched_res}_strict",
              "ngc3626": f"ngc3626_12m+7m+tp_co21_2p5kms_{matched_res}_strict",
              "ngc4435": f"ngc4435_12m+7m_co21_2p5kms_{matched_res}_strict",
              "ngc4596": f"ngc4596_12m+7m+tp_co21_2p5kms_{matched_res}_strict",
              "ngc4694": f"ngc4694_12m+7m+tp_co21_2p5kms_{matched_res}_strict",
              "ngc4697": f"ngc4697_12m_co21_2p5kms_{matched_res}_strict",
              "ngc7743": f"ngc7743_12m+7m+tp_co21_2p5kms_{matched_res}_strict",
              }

# Best reference raw frames for SDSS PSF
sdss_ref_im = {
    "ngc0524": "frame-r-007767-3-0076.fits",
    "ngc3489": "frame-r-003631-3-0323.fits",
    "ngc3599": "frame-r-005313-1-0064.fits",
    "ngc3607": "frame-r-005313-1-0067.fits",
    "ngc3626": "frame-r-005225-1-0040.fits",
    "ngc4435": "frame-r-003804-5-0194.fits",
    "ngc4457": "frame-r-001458-6-0417.fits",
    "ngc4596": "frame-r-003903-4-0059.fits",
    "ngc4694": "frame-r-003903-6-0072.fits",
    "ngc4697": "frame-r-006121-1-0124.fits",
    "ngc7743": "frame-r-007784-4-0171.fits",
}

plot_dir = os.path.join("etg_sfe", "plots")
sfr_dir = os.path.join("etg_sfe", "sfr", "rebin")
alma_dir = "alma"
sdss_dir = "sdss"
sdss_ref_dir = "sdss_ref"
sdss_psf_dir = "sdss_psf"
muse_cube_dir = "/data/beegfs/astro-storage/groups/schinnerer/PHANGS/MUSE/live/Data/ETG"
muse_r_band_dir = "muse_r_band"
psf_match_dir = "muse_psf_match"
ew_dir = os.path.join("muse", "ew")

# Set up the colourmap
colours = iter(sns.color_palette("inferno", n_colors=len(galaxies)))
