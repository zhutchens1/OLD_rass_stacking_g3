import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../codes/')
from rass_processing import mask_catl_sources


# Mask point sources
cat2rxs = pd.read_csv("cat2rxs.csv")
#mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_broad/','/srv/two/zhutchen/g3rassimages_masked_broad/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)
mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_hard/','/srv/two/zhutchen/g3rassimages_masked_hard/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)
mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_soft/','/srv/two/zhutchen/g3rassimages_masked_soft/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)
