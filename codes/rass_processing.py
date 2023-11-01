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

def measure_optimal_snr(intensity,exposure,grpcz,Rvir,H0=70.,pixel_scale=45.):
    """
    Find the optimal S/N for a stacked image by varying the
    aperture size.

    Parameters
    --------------------
    intensity : np.array
        Intensity map in units cts/sec.
    exposure : np.array
        Exposure map in units cts/sec.
    grpcz : scalar
        Redshift of group in km/s.
    Rvir : scalar
        On-sky virial radius of group in Mpc.
    H0 : scalar
        Hubble constant, default 70 (km/s)/Mpc.
    pixel_scale : scalar
        Image resolution, default 45''/px. 

    Returns
    --------------------
    snr : scalar
        Optimal S/N value.
    optimal_frac_of_Rvir : scalar
        Fraction of Rvir whose circular aperture maximized S/N.
    """
    mask = make_source_mask(intensity,nsigma=2,npixels=5,dilate_size=11)
    _, bg, _ = sigma_clipped_stats(intensity,sigma=3.0,mask=mask)
    print('Background cts/sec: ', bg)
    bg = np.zeros_like(intensity)+bg
    (A1, A2) = intensity.shape
    X,Y = np.meshgrid(np.arange(0,A1,1), np.arange(0,A2,1))
    apfrac=np.linspace(0.1,3,50)
    radius = (1*apfrac*Rvir/(grpcz/H0))*206265/pixel_scale # in px
    dist_from_center = np.sqrt((X-A1//2.)**2. + (Y-A2//2)**2.)
    snr=np.zeros_like(radius)*1.0
    for ii,RR in enumerate(radius):
        measuresel = (dist_from_center<RR)
        numerator = np.sum((intensity[measuresel]-bg[measuresel])*exposure[measuresel])
        denominator = np.sqrt(np.sum(intensity[measuresel]*exposure[measuresel]))
        snr[ii]=numerator/denominator
    optimal_frac_of_Rvir = apfrac[np.argmax(snr)]
    return np.max(snr), optimal_frac_of_Rvir

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
        mean2,_,_ = sigma_clipped_stats(image,sigma=3, maxiters=2, cenfunc = np.mean)
        sources_in_image = sources[np.where(wcs.footprint_contains(sources))]
        pixellocs = wcs.world_to_pixel(sources_in_image)
        pixellocs = np.array([pixellocs[0],pixellocs[1]]).T
        apertures = CircularAperture(pixellocs,r=apsize)
        indiv_mask=apertures.to_mask(method='center')
        final_mask=np.sum(np.array([msk.to_image(image.shape) for msk in indiv_mask]),axis=0)
        final_mask=np.where(final_mask>0, 0, 1)
        #final_mask = 1-final_mask # convert 1s to 0s, vice versa # commented and replaced with np.where statement 8/21, this was causing a bug
        newimage = image.copy()*final_mask
        print("USING MEAN AS FILL-VALUE")
        newimage[np.where(final_mask==0)]=mean2
        hdulist[0].data = newimage
        hdulist.writeto(outfiledirec+imgpath,overwrite=True)
        hdulist.close()

        #fig,axs=plt.subplots(ncols=3)
        #axs[0].imshow(image,vmin=0,vmax=5)
        #axs[1].imshow(final_mask)
        #axs[2].imshow(newimage,vmin=0,vmax=5)
        #plt.show()
        del image
        del newimage
        del hdulist

####################################################
####################################################

def mask_no_sources(infiledirec,outfiledirec,use_mp):
    countmapfiles = os.listdir(infiledirec)
    def mp_worker_func(fname):
        return mask_no_sources_worker(fname,infiledirec,outfiledirec)
    use_mp=int(use_mp)
    if use_mp==0:
        for ff in countmapfiles:
            mp_worker_func(ff)
    else:
        assert use_mp<=os.cpu_count(),"Requested more cores than available on machine."
        pool = ProcessingPool(use_mp)
        pool.map(mp_worker_func, countmapfiles)

def mask_no_sources_worker(fname,infiledirec,outfiledirec):
    hdulist=fits.open(infiledirec+fname,memmap=False)
    hdulist.writeto(outfiledirec+fname,overwrite=True)
    hdulist.close()
    del hdulist

####################################################
####################################################
####################################################
def mask_starfinder_sources(infiledirec, outfiledirec, apsize, use_mp=0):
    imagefiles = os.listdir(infiledirec)
    def mp_worker_func_(fname):
        return mask_starfinder_sources_worker(fname,infiledirec,outfiledirec,apsize)
    use_mp=int(use_mp)
    if use_mp==0:
        for ff in imagefiles:
            mp_worker_func_(ff)
    else:
        assert use_mp<=os.cpu_count(),"Requested more cores than available on machine."
        pool=ProcessingPool(use_mp)
        pool.map(mp_worker_func_, imagefiles)

def mask_starfinder_sources_worker(imgpath,imgfiledir,outfiledir,apsize):
    hdulist = fits.open(imgfiledir+imgpath, memmap=False)
    image = hdulist[0].data
    # get image stats
    scs_sigma=3
    scs_maxiters=2
    scs_cenfunc=np.mean
    mean, median, std = sigma_clipped_stats(image[image!=0],sigma=scs_sigma, maxiters=scs_maxiters, cenfunc=scs_cenfunc)
    mean2, median2, std2 = sigma_clipped_stats(image,sigma=scs_sigma, maxiters=np.max([1,scs_maxiters-1]), cenfunc=scs_cenfunc)
    # smooth image before finding sources
    smoothsigma=3.0
    if smoothsigma is not None:
        smoothimg = gaussian_filter(image, sigma=smoothsigma)
    else:
        smoothimg = np.copy(image)
    # find point sources using DAOStarFinder (photutils)
    starfinder_fwhm=3
    starfinder_threshold=5
    daofind = DAOStarFinder(fwhm=starfinder_fwhm, threshold=mean+starfinder_threshold*std)
    table = daofind.find_stars(smoothimg)
    if table is not None:
        # create and apply masks (unless it is a diffuse bright source?)
        positions=np.transpose(np.array([table['xcentroid'],table['ycentroid']]))
        apertures=CircularAperture(positions,r=apsize)
        masks=apertures.to_mask(method='center')
        # Create new image
        newimage = np.zeros_like(image)
        newmask = np.zeros_like(image)
        imagewidth=image.shape[0]
        imageheight=image.shape[1]
        newmask = np.sum(np.array([msk.to_image(shape=((imagewidth,imageheight))) for msk in masks]), axis=0)
        replacesel = np.logical_and(newmask>0,image>mean+std)
        newimage[replacesel] = mean2
        newimage[~replacesel] = image[~replacesel]
    else:
        print('skipping '+imgpath+': no point sources found')
        newimage=np.copy(image)
        final_mask=np.ones_like(newimage)
    # write to file and continue
    hdulist[0].data=newimage
    #fig,axs=plt.subplots(ncols=3)
    #axs[0].imshow(image,vmin=0,vmax=5)
    #axs[1].imshow(final_mask)
    #axs[2].imshow(newimage,vmin=0,vmax=5)
    #plt.show()
    savepath=outfiledir+imgpath#[:-5]+"_pntsourcesremoved.fits"
    hdulist.writeto(savepath, overwrite=True)
    hdulist.close()


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
        grpid=np.array(grpid)
        grpcz=np.array(grpcz)
        imagenames = np.array(os.listdir(imagefiledir))
        assert len(grpid)==len(imagenames), "Number of files in directory must match number of groups."
        print(imagenames[0].split('_'))
        imageIDs = np.array([float(imgnm.split('_')[2][3:-5]) for imgnm in imagenames])
        _, order = np.where(grpid[:,None]==imageIDs)
        imageIDs = imageIDs[order]
        imagenames = imagenames[order]
        print(imageIDs.shape)
        print(grpid.shape)
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

def average_count_rate_maps(grpid, grpcz, imagefiledir_cnt, imagefiledir_exp, stackproperty, binedges):
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
        grpid=np.array(grpid)
        grpcz=np.array(grpcz)
        imagenames = np.array(os.listdir(imagefiledir_cnt))
        imagenames_exp = np.array(os.listdir(imagefiledir_exp))
        assert len(grpid)==len(imagenames), "Number of files in directory must match number of groups."
        print(imagenames[0].split('_'))
        imageIDs = np.array([float(imgnm.split('_')[2][3:-5]) for imgnm in imagenames])
        _, order = np.where(grpid[:,None]==imageIDs)
        imageIDs = imageIDs[order]
        imagenames = imagenames[order]
        imagenames_exp = imagenames_exp[order]
        print(imageIDs.shape)
        print(grpid.shape)
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
            expmapsneeded = imagenames_exp[stacksel]
            imageIDsneeded = imageIDs[stacksel]
            groupstackID[stacksel]=i+1
            images_to_stack = [0]*len(imagenamesneeded)
            for j in range(0,len(imagenamesneeded)):
                img = imagenamesneeded[j]
                hdulist_cnt = fits.open(imagefiledir_cnt+img, memmap=False)
                cnt = hdulist_cnt[0].data
                hdulist_exp = fits.open(imagefiledir_exp+expmapsneeded[j], memmap=False)
                exp = hdulist_exp[0].data
                hdulist_cnt.close()
                hdulist_exp.close()
                images_to_stack[j]=np.array(cnt/exp)#.append(img)
                print(cnt.mean(), exp.mean())
            avg = _combine_images_worker(np.array(images_to_stack))/len(images_to_stack)
            n_in_bin.append(len(images_to_stack))
            finalimagelist.append(avg)
            print("Bin {} done.".format(i))
        return groupstackID, n_in_bin, bincenters, finalimagelist

