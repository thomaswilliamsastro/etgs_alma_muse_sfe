import os

from pymusepipe.target_sample import MusePipeSample
from pymusepipe.util_image import PointingTable


# Setting the Sample
def set_reduct(targetnames=None,
               dict_sample=None,
               ):
    if dict_sample is None:
        raise ValueError("dict_sample must be defined")

    # Data reduction for all targets
    if targetnames is not None:
        thisdic = dict((k, dict_sample[k]) for k in targetnames)
    else:
        thisdic = dict_sample
    sample = MusePipeSample(thisdic,
                            rc_filename=rc_file,
                            cal_filename=cal_file,
                            PHANGS=True,
                            )
    print("=================================")
    print("Galaxies in the sample ---")
    for i, name in enumerate(sample.targetnames):
        print("Target {0:02d}: {1:10s}".format(i + 1, name))
    print("=================================")
    return sample


# Define relevant folders (data etc)
livefold = "/data/beegfs/astro-storage/groups/schinnerer/williams/muse/ETG"
conf_folder = os.path.join(livefold, "Config")

if not os.path.exists(conf_folder):
    os.makedirs(conf_folder)

versions = [
    'dr3p2',
    'dr3p1',
]

# importing the input files
rc_file = os.path.join(conf_folder, "rc_file.dic")
cal_file = os.path.join(conf_folder, "calib_tables.dic")

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

dict_filter_for_alignment = {
    "NGC0524": "SDSS_r",
    "NGC3489": "SDSS_r",
    "NGC3599": "SDSS_r",
    "NGC3607": "SDSS_r",
    "NGC3626": "SDSS_r",
    "NGC4435": "SDSS_r",
    "NGC4457": "DUPONT_R",
    "NGC4596": "SDSS_r",
    "NGC4694": "SDSS_r",
    "NGC4697": "SDSS_r",
    "NGC7743": "SDSS_r",
}

dict_sample = {
    "NGC0524": ['ETG', {1: 1}],
    "NGC3489": ['ETG', {1: 1}],
    "NGC3599": ['ETG', {1: 1}],
    "NGC3607": ['ETG', {1: 1}],
    "NGC3626": ['ETG', {1: 1}],
    "NGC4435": ['ETG', {1: 1}],
    "NGC4457": ['ETG', {1: 1}],
    "NGC4596": ['ETG', {1: 1}],
    "NGC4694": ['ETG', {1: 1}],
    "NGC4697": ['ETG', {1: 1}],
    "NGC7743": ['ETG', {1: 1}],
}

# 1: FWHM; 2: 2.8 for noAO, 2.3 for AO; 3: Pivot wavelength

dict_psf = {
    "NGC0524": {1: ["moffat", 1.07833, 2.8, 6175, -3e-5]},
    "NGC3489": {1: ["moffat", 0.87229, 2.3, 6175, -3e-5]},
    "NGC3599": {1: ["moffat", 0.88832, 2.8, 6175, -3e-5]},
    "NGC3607": {1: ["moffat", 0.78623, 2.8, 6175, -3e-5]},
    "NGC3626": {1: ["moffat", 0.57464, 2.3, 6175, -3e-5]},
    "NGC4435": {1: ["moffat", 0.70713, 2.8, 6175, -3e-5]},
    "NGC4457": {1: ["moffat", 0.76418, 2.8, 6175, -3e-5]},
    "NGC4596": {1: ["moffat", 1.11429, 2.8, 6175, -3e-5]},
    "NGC4694": {1: ["moffat", 0.68615, 2.8, 6175, -3e-5]},
    "NGC4697": {1: ["moffat", 0.78623, 2.8, 6175, -3e-5]},
    "NGC7743": {1: ["moffat", 0.93315, 2.8, 6175, -3e-5]},
}

filter_list = [dict_filter_for_alignment[key] for key in dict_filter_for_alignment]

for targetname in target_sample:
    sample = set_reduct([targetname], dict_sample=dict_sample)

    # For whatever reason, you need a trailing / at the end here else the PointingTable
    # crashes
    offset_table_folder = os.path.join(livefold,
                                       "ETG",
                                       targetname,
                                       "Alignment/",
                                       )

    if not os.path.exists(offset_table_folder):
        os.makedirs(offset_table_folder)

    # Loop round to find the reduction version
    found_offset_table = False
    while not found_offset_table:
        for version in versions:

            offset_table_name = f"{targetname}_offset_table_{version}"
            final_offset_table_name = os.path.join(offset_table_folder,
                                                   f"{offset_table_name}.fits",
                                                   )
            if os.path.exists(final_offset_table_name):
                found_offset_table = True
            if version == versions[-1] and not found_offset_table:
                raise ValueError(f"Could not find version in any of {versions}")

    # # Reduce up to pre-align
    # sample.reduce_target_prealign(targetname,
    #                               filter_for_alignment=dict_filter_for_alignment[targetname],
    #                               name_offset_table=final_offset_table_name,
    #                               folder_offset_table=offset_table_folder,
    #                               overwrite_astropy_tables=True,
    #                               )

    pt = PointingTable(folder=offset_table_folder,
                       filtername=dict_filter_for_alignment[targetname],
                       )
    pt.qtable.write(os.path.join(offset_table_folder,
                                 f"{targetname}_pointing_table.fits",
                                 ),
                    overwrite=True,
                    )

    # Reduce up to post-align
    # sample.reduce_target_postalign(targetname,
    #                                filter_for_alignment=dict_filter_for_alignment[targetname],
    #                                name_offset_table=final_offset_table_name,
    #                                folder_offset_table=offset_table_folder,
    #                                overwrite_astropy_tables=True,
    #                                )
    #
    # # Finalise reduction
    # sample.finalise_reduction(targetname=targetname,
    #                           name_offset_table=final_offset_table_name,
    #                           folder_offset_table=offset_table_folder,
    #                           filter_for_alignment=dict_filter_for_alignment[targetname],
    #                           rot_pixtab=True,
    #                           create_pixtables=True,
    #                           create_wcs=True,
    #                           create_expocubes=True,
    #                           create_pointingcubes=True,
    #                           pointing_table=pt,
    #                           overwrite_astropy_tables=True,
    #                           verbose=True,
    #                           )

    sample.convolve_mosaic_per_pointing(targetname=targetname,
                                        pointing_table=pt,
                                        dict_psf=dict_psf[targetname],
                                        fakemode=False,
                                        best_psf=True,
                                        excluded_suffix=['copt'],
                                        )
    suffix = sample.pipes_mosaic[targetname].copt_suffix
    suffixout = f"WCS_Pall_mad_{suffix}"
    print(suffix, suffixout)
    sample.mosaic(targetname=targetname,
                  pointing_table=pt,
                  dict_psf=dict_psf[targetname],
                  suffixout=suffixout,
                  included_suffix=[suffix],
                  )

print("Complete!")
