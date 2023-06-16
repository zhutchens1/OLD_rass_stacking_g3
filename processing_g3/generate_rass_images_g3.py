import numpy as np
import pandas as pd
import sys
sys.path.insert(0,'../codes')
from make_custom_mosaics import *
from reproject import reproject_interp, reproject_exact
import multiprocessing as mp

n_cores = 35
generate_broad = True
generate_hard = False
generate_soft = False
center_defn = 'BCG'
assert center_defn=='average' or center_defn=='BCG', "Center defintion must be `average` or `BCG`."

ecofile='/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv'
resfile='/srv/one/zhutchen/g3groupfinder/resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv'
rassinputdirec='/srv/two/zhutchen/rass/'
rassoutputdirec_broad='/srv/two/zhutchen/g3rassimages_mosaics_broad_bcg/'
rassoutputdirec_hard='/srv/two/zhutchen/g3rassimages_mosaics_hard/'
rassoutputdirec_soft='/srv/two/zhutchen/g3rassimages_mosaics_soft/'
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
def mosaicfunc_broad(grpid, grpra, grpde, cmaps, emaps):
     mosaic_single_group(grpid,grpra,grpde,cmaps,emaps,512,rassoutputdirec_broad,\
            grpid.astype(str), method=reproject_exact)

def mosaicfunc_hard(grpid, grpra, grpde, cmaps, emaps):
     mosaic_single_group(grpid,grpra,grpde,cmaps,emaps,512,rassoutputdirec_hard,\
            grpid.astype(str), method=reproject_exact)

def mosaicfunc_soft(grpid, grpra, grpde, cmaps, emaps):
     mosaic_single_group(grpid,grpra,grpde,cmaps,emaps,512,rassoutputdirec_soft,\
            grpid.astype(str), method=reproject_exact)
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
eco = pd.read_csv(ecofile)
eco = eco[(eco.absrmag<=-17.33)&(eco.g3grpcz_l>3000)&(eco.g3grpcz_l<7000)&(eco.g3fc_l>0)]
res = pd.read_csv(resfile)
res = res[(res.f_b==1)&(res.absrmag<=-17.0)&(res.g3grpcz_l>4500)&(res.g3grpcz_l<7000)&(res.g3fc_l>0)]
assert min(res.g3grp_l)>max(eco.g3grp_l)

eco = eco[['name','g3grp_l','g3grpradeg_l','g3grpdedeg_l','radeg','dedeg']]
res = res[['name','g3grp_l','g3grpradeg_l','g3grpdedeg_l','radeg','dedeg']]
ecowb = pd.concat([eco,res])
print(ecowb)
del eco
del res

grpid = ecowb.g3grp_l.to_numpy()
if center_defn=='average':
    grpra = ecowb.g3grpradeg_l.to_numpy()
    grpdec = ecowb.g3grpdedeg_l.to_numpy()
elif center_defn=='BCG':
    grpra = ecowb.radeg.to_numpy()
    grpdec = ecowb.dedeg.to_numpy()

rasstable = pd.read_csv("/srv/one/zhutchen/rass_stacking_g3/codes/RASS_public_contents_lookup.csv")
names = get_neighbor_images(grpra,grpdec,rasstable.ra,rasstable.dec,rasstable.image,9)

expmaps=np.zeros_like(names,dtype='object')
cntmaps_broad=np.zeros_like(names,dtype='object')
cntmaps_hard=np.zeros_like(names,dtype='object')
cntmaps_soft=np.zeros_like(names,dtype='object')
for ii,subarr in enumerate(names):
    for jj,nm in enumerate(subarr):
        obs=nm.split('.')[0]
        cntmaps_broad[ii][jj]=rassinputdirec+obs+'/'+obs+'_im1.fits'
        cntmaps_hard[ii][jj]=rassinputdirec+obs+'/'+obs+'_im1.fits'
        cntmaps_soft[ii][jj]=rassinputdirec+obs+'/'+obs+'_im1.fits'
        expmaps[ii][jj]=rassinputdirec+obs+'/'+obs+'_mex.fits'


#make_custom_mosaics(grpid,grpra,grpdec,cntmaps,expmaps,512,rassoutputdirec,grpid.astype(str),method=reproject_exact)
if generate_broad:
    args=[grpid,grpra,grpdec,cntmaps_broad,expmaps]
    args = [tuple(x) for x in zip(*args)]
    pool=mp.Pool(n_cores)
    pool.starmap(mosaicfunc_broad,args)
    pool.close()

if generate_hard:
    args=[grpid,grpra,grpdec,cntmaps_hard,expmaps]
    args = [tuple(x) for x in zip(*args)]
    pool=mp.Pool(n_cores)
    pool.starmap(mosaicfunc_hard,args)
    pool.close()

if generate_soft:
    args=[grpid,grpra,grpdec,cntmaps_soft,expmaps]
    args = [tuple(x) for x in zip(*args)]
    pool=mp.Pool(n_cores)
    pool.starmap(mosaicfunc_soft,args)
    pool.close()

print("Don't forget! You may need to mv the outputs around in the file system so that exposure maps are in their own directory.")
