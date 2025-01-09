import glob
import os

livefold = "/data/beegfs/astro-storage/groups/schinnerer/williams/muse/ETG"
out_dir = "copt"

os.chdir(livefold)

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

for target in target_sample:

    print(target)

    filename = glob.glob(os.path.join("ETG",
                                      target,
                                      "Combined",
                                      "Cubes",
                                      "*DATACUBE_FINAL*WCS_Pall_mad*.fits"),
                         )
    if len(filename) > 1:
        raise IOError("Multiple copt files found. One of these must be wonky!")

    filename = filename[0]

    target_dir = os.path.join(out_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    out_file = os.path.join(target_dir,
                            f"{target}.fits",
                            )
    os.system(f"cp {filename} {out_file}")

print("Complete")
