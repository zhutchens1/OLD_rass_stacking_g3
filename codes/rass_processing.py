"""
RASS processing file
contains code for point source masking,
image rescaling, S/N calc etc.
"""
from scipy import ndimage
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as uu
from astropy.coordinates import SkyCoord
from photutils import DAOStarFinder, CircularAperture
from photutils.segmentation import make_source_mask
from scipy.ndimage import gaussian_filter
import os
import sys
import pickle
import gc
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
#import multiprocessing 
from pathos.multiprocessing import ProcessingPool 
from numba import jit

def mask_catl_sources(infiledirec, outfiledirec, source_ra, source_dec, apsize, use_mp=0):
    """
    Mask point sources in RASS count maps identified in external
    point source catalogs.

    in the ROSAT 2RXS point source catalog. 

    Parameters
    ------------------------
    infiledirec : str
        Read-in directory containing ROSAT FITS count maps.
    outfiledirec : str
        Write-out directory where masked FITS count maps will be written.
    source_ra : array_like
        Right ascension (J2000) of point sources in decimal degrees.
    source_dec : array_like
        Declination (J2000) of point sources in decimal degrees.
    apsize : float
        Size of apertures used in masks in pixel units (e.g., 5 pixels).
    use_mp : int
        Number of cores to use for multiprocessing. Must be <=os.cpu_count().
        If 0 (default), multiprocessing is not implemented.

    Returns
    ------------------------
    None. Masked images are saved to disk at `outfiledirec`.
    """
    countmapfiles = os.listdir(infiledirec)
    source_ra = np.array(source_ra)*uu.degree
    source_dec = np.array(source_dec)*uu.degree
    sources = SkyCoord(source_ra, source_dec)
    def mp_worker_func(fname):
        return mask_catl_sources_worker(fname,infiledirec,outfiledirec,sources,apsize)
    use_mp=int(use_mp)
    if use_mp==0:
        for ff in countmapfiles:
            mp_worker_func(ff)
    else:
        assert use_mp<=os.cpu_count(),"Requested more cores than available on machine."
        pool = ProcessingPool(use_mp)
        pool.map(mp_worker_func, countmapfiles)

def mask_catl_sources_worker(imgpath,infiledirec,outfiledirec, sources, apsize):
    if 'Cnt' in imgpath:
        print('Removing point sources from '+imgpath)
        hdulist = fits.open(infiledirec+imgpath,memmap=False)
        image = hdulist[0].data
        wcs = WCS(hdulist[0].header)
        sources_in_image = sources[np.where(wcs.footprint_contains(sources))]
        pixellocs = wcs.world_to_pixel(sources_in_image)
        pixellocs = np.array([pixellocs[0],pixellocs[1]]).T
        apertures = CircularAperture(pixellocs,r=apsize)
        indiv_mask=apertures.to_mask(method='center')
        final_mask=np.sum(np.array([msk.to_image(image.shape) for msk in indiv_mask]),axis=0)
        final_mask = 1-final_mask # convert 1s to 0s, vice versa
        newimage = image.copy()*final_mask
        hdulist[0].data = newimage
        hdulist.writeto(outfiledirec+imgpath,overwrite=True)
        hdulist.close()
        del image
        del newimage
        del hdulist


####################################################
####################################################
####################################################
def scale_crop_images(imagefiledir, outfiledir, rafiledict, defiledict, czfiledict, crop=False, crop_window_size=None, imwidth=300, res=45, H0=70., use_mp=0, progressConf=False):
    """
    Scale images to a common redshift. Images must be square.
    
    Parameters
    -------------
    imgfiledir : str
        Path to directory containing input images for scaling.
        Each FITS file in this directory must be named consistently
        with the rest of this program (e.g. RASS-Int_Broad_grp13_ECO03822.fits).
    outfiledir : str
        Path where scaled images should be written.
    rafiledict : dict
        Dictionary mapping filenames (index) to group RAs (values).
    defiledict : dict
        Dictionary mapping filenames (index) to group DECs (values).
    czfiledict : dict
        Dictionary mapping filenames (index) to group redshifts (values).
    crop : bool, default False
        If True, all images are cropped to size of the smallest scaled images.
        This may preferable if the zero-padding on image borders will produce
        artefacts in other analysis --- cropping maintains uniform noise
        between each output image, but all images are resized according to the
        largest and smallest cz values.
    crop_window_size : int
        Size of cropping window. If None (default) and crop is True, then the size
        will be determined using the image image width, resolution, and max cz value.
        If passed as float, value will be casted as int.
    imwidth : int
        Width of image, assumed to be square, in pixels; default 300.
    res : float
        Pixel resolution in arcseconds, default 45.
    H0 : float
        Hubble constant in km/s/(Mpc), default 70.
    use_mp = 0
        Number of cores for parallel processing, must be <=os.cpu_count().
        If 0 (default), multiprocessing is not used. 
    progressConf : bool, default False
        If True, the loop prints out a progress statement when each image is finished.

    Returns
    -------------
    Scaled/subtracted images are written to the specified path.
    """
    imagenames = np.array(os.listdir(imagefiledir))
    #imageIDs = np.array([float(imgnm.split('_')[2][3:-5]) for imgnm in imagenames])
    czvals = np.array(list(czfiledict.values()))
    czmin=np.min(czvals)
    czmax=np.max(czvals)
    if crop and (crop_window_size is None): # work out what area to retain
        D1 = (imwidth*res/206265)*(czmin/H0)
        Npx = int((D1*H0)/czmax * (206265/res))
        Nbound = (imwidth-Npx)//2 # number of pixels spanning border region
        crop_window_size = int(imwidth - 2*Nbound)
    else:
        crop_window_size = int(crop_window_size)
    loop_worker = lambda ii: _scaler_worker_func(ii,crop_window_size,czmax,imagenames,imagefiledir,outfiledir,rafiledict,defiledict,czfiledict,crop,imwidth,res,H0,progressConf)
    if use_mp==0:
        for kk in range(0,len(imagenames)):
            loop_worker(kk)
    else:
        assert use_mp<=os.cpu_count(),'Requesting more cores than available on machine.'
        pool = ProcessingPool(use_mp)
        pool.map(loop_worker, [kk for kk in range(0,len(imagenames))])
        

def _scaler_worker_func(index_,crop_window_size,czmax,imagenames,imagefiledir,outfiledir,rafiledict,defiledict,czfiledict,crop,imwidth,res,H0,progressConf):
        hdulist = fits.open(imagefiledir+imagenames[index_], memap=False)
        img = hdulist[0].data
        #czsf = self.grpcz[self.grpid==imageIDs[k]]/czmax
        czsf = czfiledict[imagenames[index_]]/czmax
        #if len(czsf)==0:
        #    continue
        img = ndimage.geometric_transform(img, scale_image, cval=0, extra_keywords={'scale':czsf, 'imwidth':imwidth})
        if crop: # work out which pixels to retain
            hdulist[0].data = img
            wcs = WCS(hdulist[0].header)
            #sel=(self.grpid==imageIDs[k])
            croppedim = Cutout2D(img, SkyCoord(rafiledict[imagenames[index_]]*uu.degree,defiledict[imagenames[index_]]*uu.degree,frame='fk5'), crop_window_size, wcs)
            hdu = fits.PrimaryHDU(croppedim.data, header=croppedim.wcs.to_header())
            hdulist=fits.HDUList([hdu])
        else:
            hdulist[0].data = img
        hdulist.writeto(outfiledir+imagenames[index_], overwrite=True)
        hdulist.close()
        if progressConf: print("Finished scaling {}.".format(imagenames[index_]))

def scale_image(output_coords,scale,imwidth):
    mid = imwidth//2
    return (output_coords[0]/scale+mid-mid/scale, output_coords[1]/scale+mid-mid/scale)


####################################################
####################################################
####################################################
# Image stacking
####################################################
####################################################
####################################################

def stack_images(grpid, grpcz, imagefiledir, stackproperty, binedges):
        """
        Stack X-ray images of galaxy groups in bins of group properties
        (e.g. richness or halo mass).

        Parameters
        --------------------
        imgfiledir : str
            Path to directory containing input images for stacking.
            Each FITS file in this directory must be named consistently
            with the rest of this program (e.g. RASS-Int_Broad_grp13_ECO03822.fits).
        stackproperty : iterable
            Group property to be used for binning (e.g. halo mass). This
            list should include an entry for *every* group, as to match
            the length of self.grpid.
        bins : iterable
            Array of bins for stacking. It should represent the bin *edges*. 
            Example: if bins=[11,12,13,14,15,16], then the resulting bins
            are [11,12], [12,13], [13,14], [14,15], [15,16].
        
        Returns
        --------------------
        groupstackID : np.array
            The ID of stack to which each galaxy group in self.grpid belongs.
        n_in_bin : list
            Number of images contributing to each stack.
        bincenters : np.array
            Center of each bin used in stacking.
        finalimagelist : list
            List of final stacked images, i.e. finalimagelist[i] is a sigma-clipped
            average of all images whose groups satisifed the boudnary conditions of
            bin[i]. Each image is a 2D numpy array.
        """
        imagenames = np.array(os.listdir(imagefiledir))
        assert len(grpid)==len(imagenames), "Number of files in directory must match number of groups."
        print(imagenames[0].split('_'))
        imageIDs = np.array([float(imgnm.split('_')[2][3:-5]) for imgnm in imagenames])
        _, order = np.where(grpid[:,None]==imageIDs)
        imageIDs = imageIDs[order]
        imagenames = imagenames[order]
        assert (imageIDs==grpid).all(), "ID numbers are not sorted properly."

        stackproperty = np.asarray(stackproperty)
        groupstackID = np.zeros_like(stackproperty)        
        binedges = np.array(binedges)
        leftedges = binedges[:-1]
        rightedges = binedges[1:]
        bincenters = (leftedges+rightedges)/2.
        finalimagelist = []
        n_in_bin=[]
        for i in range(0,len(bincenters)):
            stacksel = np.where(np.logical_and(stackproperty>=leftedges[i], stackproperty<rightedges[i]))
            imagenamesneeded = imagenames[stacksel]
            imageIDsneeded = imageIDs[stacksel]
            groupstackID[stacksel]=i+1
            images_to_stack = [0]*len(imagenamesneeded)
            for j in range(0,len(imagenamesneeded)):
                img = imagenamesneeded[j]
                hdulist = fits.open(imagefiledir+img, memmap=False)
                img = hdulist[0].data
                if np.isnan(img).all(): print('all NANs: ', imagenamesneeded[j])
                hdulist.close()
                images_to_stack[j]=np.array(img)#.append(img)
            avg = _combine_images_worker(np.array(images_to_stack))#np.sum(images_to_stack,axis=0)#/len(images_to_stack)
            n_in_bin.append(len(images_to_stack))
            finalimagelist.append(avg)
            print("Bin {} done.".format(i))
        return groupstackID, n_in_bin, bincenters, finalimagelist

@jit(nopython=True)
def _combine_images_worker(image_arr):
    return np.sum(image_arr,axis=0)
