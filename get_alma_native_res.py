import os

from astropy.io import fits

from vars import wisdom_dir, alma_dir, galaxies

# ALMA file mappings
alma_files = {"ngc0524": f"ngc0524_12m+7m_co21_2p5kms",
              "ngc3489": f"ngc3489_12m+7m+tp_co21_2p5kms",
              "ngc3599": f"ngc3599_12m+7m+tp_co21_2p5kms",
              "ngc3607": f"ngc3607_12m+7m_co21_2p5kms",
              "ngc3626": f"ngc3626_12m+7m+tp_co21_2p5kms",
              "ngc4435": f"ngc4435_12m+7m_co21_2p5kms",
              "ngc4596": f"ngc4596_12m+7m+tp_co21_2p5kms",
              "ngc4694": f"ngc4694_12m+7m+tp_co21_2p5kms",
              "ngc4697": f"ngc4697_12m_co21_2p5kms",
              "ngc7743": f"ngc7743_12m+7m+tp_co21_2p5kms",
              }

os.chdir(wisdom_dir)

for galaxy in galaxies:

    hdu_name = os.path.join(alma_dir, galaxy, f"{alma_files[galaxy]}.fits")

    with fits.open(hdu_name) as hdu:
        bmaj = hdu[0].header["BMAJ"]
        bmaj *= 3600

        print(f"{galaxy} native res: {bmaj:.2f}arcsec")
