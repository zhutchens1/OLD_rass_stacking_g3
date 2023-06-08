import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../codes/')
from rass_processing import mask_catl_sources, scale_crop_images

ecofile='/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv'
resfile='/srv/one/zhutchen/g3groupfinder/resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv'
eco = pd.read_csv(ecofile)
eco = eco[(eco.absrmag<=-17.33)&(eco.g3grpcz_l>3000)&(eco.g3grpcz_l<7000)&(eco.g3fc_l>0)]
res = pd.read_csv(resfile)
res = res[(res.f_b==1)&(res.absrmag<=-17.0)&(res.g3grpcz_l>4500)&(res.g3grpcz_l<7000)&(res.g3fc_l>0)]
assert min(res.g3grp_l)>max(eco.g3grp_l)

eco = eco[['name','g3grp_l','g3grpradeg_l','g3grpdedeg_l','radeg','dedeg','g3grpcz_l']]
res = res[['name','g3grp_l','g3grpradeg_l','g3grpdedeg_l','radeg','dedeg','g3grpcz_l']]
ecowb = pd.concat([eco,res])
del eco
del res


####################################
####################################
####################################
# Mask point sources
#cat2rxs = pd.read_csv("cat2rxs.csv")
#mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_broad/','/srv/two/zhutchen/g3rassimages_masked_broad/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)
#mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_hard/','/srv/two/zhutchen/g3rassimages_masked_hard/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)
#mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_soft/','/srv/two/zhutchen/g3rassimages_masked_soft/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)
#mask_catl_sources('/srv/two/zhutchen/g3rassimages_mosaics_broad_bcg/','/srv/two/zhutchen/g3rassimages_masked_broad_bcg/',cat2rxs.RA_DEG,cat2rxs.DEC_DEG,5,use_mp=30)

files = os.listdir("/srv/two/zhutchen/g3rassimages_mosaics_broad/")
czdict = {ff : float(ecowb['g3grpcz_l'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
radict = {ff : float(ecowb['g3grpradeg_l'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
dedict = {ff : float(ecowb['g3grpdedeg_l'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
scale_crop_images('/srv/two/zhutchen/g3rassimages_masked_broad/','/srv/two/zhutchen/g3rassimages_scaled_broad/',radict,dedict,czdict,crop=True,imwidth=512,progressConf=True,use_mp=30)
