import os

from TardisPipeline import MainPipeline

outpath = '/data/beegfs/astro-storage/groups/schinnerer/williams/muse/ETG/DAP'

configfile = os.path.join(outpath, 'config.ini')

MainPipeline.run_all(path=configfile)

print("Complete!")
