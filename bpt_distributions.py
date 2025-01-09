import copy
import os

import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import seaborn as sns

from vars import wisdom_dir, galaxies, plot_dir, sfr_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes("deep")

os.chdir(wisdom_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

bpt_names = [
    "sf",
    "comp",
    "sy",
    "lier",
]
bpt_colour = {"sf": "b",
              "comp": "y",
              "sy": "g",
              "lier": "r",
              }
bpt_label = {"sf": "SF",
             "comp": "Comp.",
             "sy": "Sy",
             "lier": "LIER",
             }

bpt_dict = {}

for name in bpt_names:
    bpt_dict[name] = []

for galaxy in galaxies:
    bpt_filename = os.path.join(sfr_dir,
                                f"{galaxy}_sfr_maps.fits",
                                )

    with fits.open(bpt_filename) as hdu:
        data = copy.deepcopy(hdu["BPT"].data)
        total_n = len(data[np.isfinite(data)])

        for i in range(len(bpt_names)):
            n = len(data[data == i])
            bpt_dict[bpt_names[i]].append(n / total_n)

plt.figure(figsize=(5, 4))
plt.hist([bpt_dict[key] for key in bpt_dict],
         bins=10,
         range=[0, 1],
         histtype='stepfilled',
         stacked=True,
         color=[bpt_colour[name] for name in bpt_names],
         label=[bpt_label[name] for name in bpt_names],
         )

plt.grid()

plt.xlim(0, 1)

plt.xlabel(r"$f_\mathrm{BPT~Class}$")
plt.ylabel(r"$N$")

plt.legend(loc='upper right',
           fancybox=False,
           edgecolor='k',
           )

plt.tight_layout()

plt.show()

print('Complete!')
