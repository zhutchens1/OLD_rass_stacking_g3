import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs
from reproject import reproject_interp, reproject_exact
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as uu
import matplotlib.pyplot as plt

def make_custom_mosaics(groupid, groupra, groupdec, count_paths, exp_paths, outsz, outdir, savehandle, method):
    """
    Make custom images of galaxy groups at specified RA/Dec using mosaics of raw images.

    Parameters
    --------------------
    groupra : iterable
        RA values where images should be extracted, decimal degrees.
    groupdec : iterable
        Dec values where images should be extracted, decimal degrees.
    countpaths : iterable
        Sequence of array_like objects containing the filenames of the closest
        count maps to each group, e.g. as returned from get_neighbor_images.
    exp_paths : iterable
        Sequence of array_like objects containing the filenames of the closest
        exposure maps to each group, e.g. as returned from get_neighbor_images.
        These should respond one-to-one with the counts maps in `countpaths`.
    outsz : scalar
        Output size of extracted, custom image. The final image will be square,
        e.g. outsz=500 will return a 500 x 500 image.
    outdir : str
        Directory where output images will be written.
    savehandle : array-like, str elements
        Identifier of the image in final filename, e.g. group ID and or central galaxy name.
        Example: if savehandle='10830', then the file will be saved as RASS-Cnt_Broad_grp10830.fits
        in outdir. Length matches groupra, groupdec
    method : callable
        Method of resampling pixels, must be either reproject.reproject_interp
        or reproject.reproject_exact.

    Returns
    -----------------------
    None. Images are mosaicked, extracted, and written to disk. 
    """
    assert callable(method), "Parameter `method` must reproject.reproject_interp or reproject.reproject_exact."
    groupid=np.array(groupid)
    groupra=np.array(groupra)
    groupdec=np.array(groupdec)
    coords=SkyCoord(ra=groupra*uu.degree, dec=groupdec*uu.degree)
    count_paths=np.array(count_paths,dtype=object)
    exp_paths=np.array(exp_paths,dtype=object)
    
    for ii,gg in enumerate(groupid):
        cname=count_paths[ii]
        ename=exp_paths[ii]
        chdus = [fits.open(cm)[0] for cm in cname]
        ehdus = [fits.open(em)[0] for em in ename]
        wcs_out,shape_out = find_optimal_celestial_wcs(chdus)
        cmosaic, cfp = reproject_and_coadd(chdus,output_projection=wcs_out,shape_out=shape_out,reproject_function=method)
        emosaic, efp = reproject_and_coadd(ehdus,output_projection=wcs_out,shape_out=shape_out,reproject_function=method)
        extract_write_from_mosaic(cmosaic,coords[ii],wcs_out,outsz,outdir+"RASS-Cnt_Broad_grp"+savehandle[ii]+".fits")
        extract_write_from_mosaic(emosaic,coords[ii],wcs_out,outsz,outdir+"RASS-Exp_Broad_grp"+savehandle[ii]+".fits")

def mosaic_single_group(groupid, groupra, groupdec, count_paths, exp_paths, outsz, outdir, savehandle, method):
    cname=count_paths
    ename=exp_paths
    coords=SkyCoord(ra=groupra*uu.degree,dec=groupdec*uu.degree)
    chdus = [fits.open(cm)[0] for cm in cname]
    ehdus = [fits.open(em)[0] for em in ename]
    wcs_out,shape_out = find_optimal_celestial_wcs(chdus)
    cmosaic, cfp = reproject_and_coadd(chdus,output_projection=wcs_out,shape_out=shape_out,reproject_function=method)
    emosaic, efp = reproject_and_coadd(ehdus,output_projection=wcs_out,shape_out=shape_out,reproject_function=method)
    extract_write_from_mosaic(cmosaic,coords,wcs_out,outsz,outdir+"RASS-Cnt_Broad_grp"+savehandle+".fits")
    extract_write_from_mosaic(emosaic,coords,wcs_out,outsz,outdir+"RASS-Exp_Broad_grp"+savehandle+".fits")


def extract_write_from_mosaic(mosaic,position,wcs,outsz,savepath):
    """
    Extract a 2D cutout from a mosaic image.

    Parameters
    ----------------------------
    mosaic : np.array
        Numpy array representing the mosaic.
    position : astropy.coordinates.SkyCoord instance
        On-sky position at which to perform the extraction.
    wcs : astropy.WCS instance
        World coordinate system of the mosaic.
    outsz : int
        Size of image to be extracted and written.
        (e.g., 500 x 500 if outsz=500).
    savepath : str
        Location where cutout image should be saved.
    """
    image = Cutout2D(mosaic,position=position,wcs=wcs,size=outsz)
    hdu = fits.PrimaryHDU(image.data, header=image.wcs.to_header())
    hdulist=fits.HDUList([hdu])
    hdulist.writeto(savepath,overwrite=True)

def get_neighbor_images(groupra, groupdec, imagera, imagedec, imagename, kk=9):
    """
    Given a set of galaxies or galaxy groups, and a separate set of 
    images, find the k neighboring images surrounding each group (incl.
    the image that contains the group of interest.) This algoritm implements
    sklearn.neighbors.KDTree to find images nearby to groups.

    Parameters
    ------------------------
    groupra : iterable
        RA of group centers, size N, where N is the number
        of unique groups in the catalog. Decimal degrees.
    groupdec : iterable
        Declination of group centers, size N. Decimal degrees.
    imagera : iterable
        RA of image centers, size p. Decimal degrees.
    imagedec : iterable
        Declination of image centers, size p. Decimal degrees.
    imagename : iterable
        Array of identificiation values for each image. Can be strings,
        int, etc. Size p.
    kk : int, default 9
        Number of images to return for each group. This argument is
        passed as `k` to sklearn.neighbors.KDTree.
        
    Returns
    -------------------------
    neighbors : np.array of shape (N,kk)
        Matrix containing the kk-nearest images, denoted using values
        rom `imagename`, for each observation in the group* dataset.
    """
    groupra=np.array(groupra)
    groupdec=np.array(groupdec)
    imagera=np.array(imagera)
    imagedec=np.array(imagedec)
    dtr=np.pi/180.
    imageX = np.sin(np.pi/2.-imagedec*dtr)*np.cos(imagera*dtr) 
    imageY = np.sin(np.pi/2.-imagedec*dtr)*np.sin(imagera*dtr)
    imageZ = np.cos(np.pi/2.-imagedec*dtr)
    groupX = np.sin(np.pi/2.-groupdec*dtr)*np.cos(groupra*dtr) 
    groupY = np.sin(np.pi/2.-groupdec*dtr)*np.sin(groupra*dtr)
    groupZ = np.cos(np.pi/2.-groupdec*dtr)
    imagename=np.array(imagename,dtype=object)
    XX = np.array([imageX,imageY,imageZ])
    tree = KDTree(np.array(XX.T))
    groupdata = np.array([groupX,groupY,groupZ])
    idx = tree.query(groupdata.T, kk, return_distance=False)
    neighbors = imagename[idx]
    return neighbors


def mosaicfunc(grpid, grpra, grpde, cmaps, emaps):
     mosaic_single_group(grpid,grpra,grpde,cmaps,emaps,512,'./testimages/',\
            grpid.astype(str), method=reproject_exact)

if __name__=='__main__':
    eco = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
    eco = eco[eco.g3fc_l==1]
 
    rasstable = pd.read_csv("/srv/one/zhutchen/rass_stacking_g3/RASS_public_contents_lookup.csv")
    econame=np.array(eco.name,dtype=object)
    names=get_neighbor_images(eco.g3grpradeg_l, eco.g3grpdedeg_l, rasstable.ra, rasstable.dec, rasstable.image, 5)

    exposuremaps=np.zeros_like(names,dtype='object')
    countmaps=np.zeros_like(names,dtype='object')
    for ii,subarr in enumerate(names):
        for jj,name in enumerate(subarr):
            obs=name.split('.')[0]
            countmaps[ii][jj]='../rass/'+obs+'/'+obs+'_im1.fits'
            exposuremaps[ii][jj]='../rass/'+obs+'/'+obs+'_mex.fits'

    grpid = np.array(eco.g3grp_l)
    grpra = np.array(eco.g3grpradeg_l)
    grpde = np.array(eco.g3grpdedeg_l)
    import time

    use_mp=1
    tt=time.time()
    if use_mp:
        import multiprocessing
        sel = np.where(grpid==14.)
        args=[grpid[sel],grpra[sel],grpde[sel],countmaps[sel],exposuremaps[sel]]
        args = [tuple(x) for x in zip(*args)]
        print(args)
        #mosaicfunc(*args[0])
        pool=multiprocessing.Pool()
        pool.starmap(mosaicfunc,args)
    print(time.time()-tt)
