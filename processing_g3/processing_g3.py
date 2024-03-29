import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../codes/')
from rass_processing import mask_catl_sources, scale_crop_images, stack_images, mask_starfinder_sources, mask_no_sources, average_count_rate_maps
from datetime import datetime
import pickle

today = datetime.today().strftime('%m%d%Y')

number_of_cores = 40 # if 0, multiprocessing is not implemented.
filters = ['_broad/']#['_hard/','_soft/'] # comment any of these out to not do them. # each element should be '_filter/' (file hierarchy)
file_arch_mosaic = '/srv/two/zhutchen/g3rassimages_mosaics' # no trailing / because will be added to elements of `filters`
file_arch_masked = '/srv/two/zhutchen/g3rassimages_masked'
file_arch_scaled = '/srv/two/zhutchen/g3rassimages_scaled'
expmap_direc = '/srv/two/zhutchen/g3rassimages_exposuremaps/'
expmap_scaled_direc = '/srv/two/zhutchen/g3rassimages_exposuremaps_scaled/'
stackingoutput=['/srv/one/zhutchen/rass_stacking_g3/stackedimages/stackingresult'+flt[:-1]+'_'+today+'.pkl' for flt in filters]
do_masking = True 
do_scaling = True
do_stacking = True 
#####################################
ecofile='/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv'
resfile='/srv/one/zhutchen/g3groupfinder/resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv'
eco = pd.read_csv(ecofile)
eco = eco[(eco.absrmag<=-17.33)&(eco.g3grpcz_l>3000)&(eco.g3grpcz_l<7000)&(eco.g3fc_l>0)]
res = pd.read_csv(resfile)
res = res[(res.f_b==1)&(res.absrmag<=-17.0)&(res.g3grpcz_l>4500)&(res.g3grpcz_l<7000)&(res.g3fc_l>0)]
assert min(res.g3grp_l)>max(eco.g3grp_l)

eco = eco[['name','g3grp_l','g3grpradeg_l','g3grpdedeg_l','radeg','dedeg','g3grpcz_l','g3logmh_l','cz']]
res = res[['name','g3grp_l','g3grpradeg_l','g3grpdedeg_l','radeg','dedeg','g3grpcz_l','g3logmh_l','cz']]
ecowb = pd.concat([eco,res])
del eco
del res

print("WARNING: This code not functioning correctly")
if filters[0].endswith('bcg/'):
    ecowb.loc[:,'grpra_for_stacking'] = ecowb.radeg
    ecowb.loc[:,'grpde_for_stacking'] = ecowb.dedeg
    ecowb.loc[:,'grpcz_for_stacking'] = ecowb.cz
else:
    ecowb.loc[:,'grpra_for_stacking'] = ecowb.g3grpradeg_l
    ecowb.loc[:,'grpde_for_stacking'] = ecowb.g3grpdedeg_l
    ecowb.loc[:,'grpcz_for_stacking'] = ecowb.g3grpcz_l
####################################
####################################
####################################
# Mask point sources
if do_masking:
    cat2rxs = pd.read_csv("cat2rxs.csv")
    cat2rxs = cat2rxs[cat2rxs.EXI_ML>9]
    for filt in filters:
        #mask_catl_sources(file_arch_mosaic+filt, file_arch_masked+filt,cat2rxs.RA_DEG,cat2rxs.DEC_DEG,6,use_mp=number_of_cores)
        mask_starfinder_sources(file_arch_mosaic+filt, file_arch_masked+filt, 6, use_mp=number_of_cores)
        #mask_no_sources(file_arch_mosaic+filt, file_arch_masked+filt, use_mp=number_of_cores)

####################################
####################################
####################################
# image rescaling
crsize=256 # cropping size
if do_scaling:
    files = os.listdir("/srv/two/zhutchen/g3rassimages_mosaics_broad/")
    czdict = {ff : float(ecowb['grpcz_for_stacking'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
    radict = {ff : float(ecowb['grpra_for_stacking'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
    dedict = {ff : float(ecowb['grpde_for_stacking'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
    for filt in filters:
        scale_crop_images(file_arch_masked+filt,file_arch_scaled+filt,radict,dedict,czdict,crop=True,crop_window_size=crsize,imwidth=512,progressConf=True,use_mp=30)


files = os.listdir(expmap_direc)
czdict = {ff : float(ecowb['grpcz_for_stacking'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
radict = {ff : float(ecowb['grpra_for_stacking'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
dedict = {ff : float(ecowb['grpde_for_stacking'][(ecowb.g3grp_l==float(ff.split('_')[2][3:-5]))].values) for ff in files}
scale_crop_images(expmap_direc,expmap_scaled_direc,radict,dedict,czdict,crop=True,crop_window_size=crsize,imwidth=512,progressConf=False,use_mp=30)

#####################################
#####################################
#####################################
# Stack images

if do_stacking:
    for ii,filt in enumerate(filters):
        stID,nb,binc,cmap=stack_images(ecowb.g3grp_l.to_numpy(),ecowb.grpcz_for_stacking.to_numpy(),file_arch_scaled+filt,ecowb.g3logmh_l.to_numpy(),[11.,12.1,13.3,14.3,14.5])
        _,_,_,emap=stack_images(ecowb.g3grp_l.to_numpy(),ecowb.grpcz_for_stacking.to_numpy(),expmap_scaled_direc,ecowb.g3logmh_l.to_numpy(),[11.,12.1,13.3,14.3,14.5])
        result = [stID,nb,binc,cmap,emap]
        pickle.dump(result,open(stackingoutput[ii],'wb')) 


        stID,nb,binc,rate_maps=average_count_rate_maps(ecowb.g3grp_l.to_numpy(),ecowb.grpcz_for_stacking.to_numpy(),file_arch_scaled+filt,expmap_scaled_direc,ecowb.g3logmh_l.to_numpy(),[11.,12.1,13.3,14.3,14.5,15])
        result=[stID,nb,binc,rate_maps]
        pickle.dump(result,open(stackingoutput[ii][:-4]+'_averagecrmap.pkl','wb'))
