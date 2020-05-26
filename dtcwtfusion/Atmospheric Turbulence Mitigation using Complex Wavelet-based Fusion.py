# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:20:27 2020

@author: SYunMoon

Align, register and fuse frames specified on the command line using the
DTCWT fusion algorithm.

"""

#from __future__ import print_function, division, absolute_import

import logging

import dtcwt
import dtcwt.registration
import dtcwt.sampling
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from six.moves import xrange
from skimage import io
from matplotlib import pyplot as plt
import cv2


def align(frames, template):
    """
    Warp each slice of the 3D array frames to align it to *template*.

    """
    if frames.shape[:2] != template.shape:
        raise ValueError('Template must be same shape as one slice of frame array')

    # Calculate xs and ys to sample from one frame
    xs, ys = np.meshgrid(np.arange(frames.shape[1]), np.arange(frames.shape[0]))

    # Calculate window to use in FFT convolve
    w = np.outer(np.hamming(template.shape[0]), np.hamming(template.shape[1]))

    # Calculate a normalisation for the cross-correlation
    ccnorm = 1.0 / fftconvolve(w, w)

    # Set border of normalisation to zero to avoid overfitting. Borser is set so that there
    # must be a minimum of half-frame overlap
    ccnorm[:(template.shape[0]>>1),:] = 0
    ccnorm[-(template.shape[0]>>1):,:] = 0
    ccnorm[:,:(template.shape[1]>>1)] = 0
    ccnorm[:,-(template.shape[1]>>1):] = 0

    # Normalise template
    tmpl_min = template.min()
    norm_template = template - tmpl_min
    tmpl_max = norm_template.max()
    norm_template /= tmpl_max

    warped_ims = []
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Aligning frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]

        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max

        # Convolve template and frame
        conv_im = fftconvolve(norm_template*w, np.fliplr(np.flipud(norm_frame*w)))
        conv_im *= ccnorm

        # Find maximum location
        max_loc = np.unravel_index(conv_im.argmax(), conv_im.shape)

        # Convert location to shift
        dy = max_loc[0] - template.shape[0] + 1
        dx = max_loc[1] - template.shape[1] + 1
        logging.info('Offset computed to be ({0},{1})'.format(dx, dy))

        # Warp image
        warped_ims.append(dtcwt.sampling.sample(frame, xs-dx, ys-dy, method='bilinear'))

    return np.dstack(warped_ims)

def register(frames, template, nlevels=7):
    """
    Use DTCWT registration to return warped versions of frames aligned to template.

    """
    # Normalise template
    tmpl_min = template.min()
    norm_template = template - tmpl_min
    tmpl_max = norm_template.max()
    norm_template /= tmpl_max

    # Transform template
    transform = dtcwt.Transform2d()
    template_t = transform.forward(norm_template, nlevels=nlevels)

    warped_ims = []
    for frame_idx in xrange(frames.shape[2]):
        logging.info('Registering frame {0}/{1}'.format(frame_idx+1, frames.shape[2]))
        frame = frames[:,:,frame_idx]

        # Normalise frame
        norm_frame = frame - tmpl_min
        norm_frame /= tmpl_max

        # Transform frame
        frame_t = transform.forward(norm_frame, nlevels=nlevels)

        # Register
        reg = dtcwt.registration.estimatereg(frame_t, template_t)
        warped_ims.append(dtcwt.registration.warp(frame, reg, method='bilinear'))

    return np.dstack(warped_ims)

def tonemap(array):
    # The normalisation strategy here is to let the middle 98% of
    # the values fall in the range 0.01 to 0.99 ('black' and 'white' level).
    black_level = np.percentile(array,  1)
    white_level = np.percentile(array, 99)

    norm_array = array - black_level
    norm_array /= (white_level - black_level)
    norm_array = np.clip(norm_array + 0.01, 0, 1)

    return np.array(norm_array * 255, dtype=np.uint8)

def save_image(filename, array):
    # Copy is workaround for http://goo.gl/8fuOJA
    im = Image.fromarray(tonemap(array).copy(), 'L')

    logging.info('Saving "{0}"'.format(filename + '.png'))
    im.save(filename + '.png')

def transform_frames(frames, nlevels=9):
    # Transform each registered frame storing result
    lowpasses = []
    highpasses = []
    for idx in xrange(nlevels):
        highpasses.append([])

    transform = dtcwt.Transform2d()
    for frame_idx in xrange(frames.shape[2]):
        
        frame = frames[:,:,frame_idx]
        frame_t = transform.forward(frame, nlevels=nlevels)

        lowpasses.append(frame_t.lowpass)
        for idx in xrange(nlevels):
            highpasses[idx].append(frame_t.highpasses[idx][:,:,:,np.newaxis])

    return np.dstack(lowpasses), tuple(np.concatenate(hp, axis=3) for hp in highpasses)

def reconstruct(lowpass, highpasses):
    transform = dtcwt.Transform2d()
    t = dtcwt.Pyramid(lowpass, highpasses)
    return transform.inverse(t)

def shrink_coeffs(highpasses):
    """Implement Bivariate Laplacian shrinkage as described in [1].
    *highpasses* is a sequence containing wavelet coefficients for each level
    fine-to-coarse. Return a sequence containing the shrunken coefficients.

    [1] A. Loza, D. Bull, N. Canagarajah, and A. Achim, “Non-gaussian model-
    based fusion of noisy frames in the wavelet domain,” Comput. Vis. Image
    Underst., vol. 114, pp. 54–65, Jan. 2010.

    """
    shrunk_levels = []

    # Estimate noise from first level coefficients:
    # \sigma_n = MAD(X_1) / 0.6745

    # Compute median absolute deviation of wavelet magnitudes. This is more than
    # a little magic compared to the 1d version.
    level1_mad_real = np.median(np.abs(highpasses[0].real - np.median(highpasses[0].real)))
    level1_mad_imag = np.median(np.abs(highpasses[0].imag - np.median(highpasses[0].imag)))
    sigma_n = np.sqrt(level1_mad_real*level1_mad_real + level1_mad_imag+level1_mad_imag) / (np.sqrt(2) * 0.6745)

    # In this context, parent == coarse, child == fine. Work from
    # coarse to fine
    shrunk_levels.append(highpasses[-1])
    for parent, child in zip(highpasses[-1:0:-1], highpasses[-2::-1]):
        # We will shrink child coefficients.

        # Rescale parent to be the size of child
        parent = dtcwt.sampling.rescale(parent, child.shape[:2], method='nearest')

        # Construct gain for shrinkage separately per direction and for real and imag
        real_gain = np.ones_like(child.real)
        imag_gain = np.ones_like(child.real)
        for dir_idx in xrange(parent.shape[2]):
            child_d = child[:,:,dir_idx]
            parent_d = parent[:,:,dir_idx]

            # Estimate sigma_w and gain for real
            real_sigma_w = np.sqrt(np.maximum(1e-8, np.var(child_d.real) - sigma_n*sigma_n))
            real_R = np.sqrt(parent_d.real*parent_d.real + child_d.real*child_d.real)
            real_gain[:,:,dir_idx] = np.maximum(0, real_R - (np.sqrt(3)*sigma_n*sigma_n)/real_sigma_w) / real_R

            # Estimate sigma_w and gain for imag
            imag_sigma_w = np.sqrt(np.maximum(1e-8, np.var(child_d.imag) - sigma_n*sigma_n))
            imag_R = np.sqrt(parent_d.imag*parent_d.imag + child_d.imag*child_d.imag)
            imag_gain[:,:,dir_idx] = np.maximum(0, imag_R - (np.sqrt(3)*sigma_n*sigma_n)/imag_sigma_w) / imag_R

        # Shrink child levels
        shrunk = (child.real * real_gain) + 1j * (child.imag * imag_gain)
        shrunk_levels.append(shrunk)

    return shrunk_levels[::-1]

import glob

input_frames = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))

for filename in glob.glob('../image/temp2/*.bmp'): 
    
    image=io.imread(filename)
    img = clahe.apply(image)
    
    img = img - img.min()
    img = img / img.max()
    
    input_frames.append(img)

input_frames = np.dstack(input_frames)

#    if ref_strategy == 'middle':
#        reference_frame = input_frames[:,:,input_frames.shape[2]>>1]
#    elif ref_strategy == 'first':
#        reference_frame = input_frames[:,:,0]
#    elif ref_strategy == 'last':
#        reference_frame = input_frames[:,:,-1]
        
# ref_strategy == 'max-mean':
means = np.array(list(np.mean(input_frames[:,:,idx]) for idx in xrange(input_frames.shape[2])))
reference_frame = input_frames[:,:,means.argmax()]
#    elif ref_strategy == 'max-range':
#        ranges = np.array(list(input_frames[:,:,idx].max() - input_frames[:,:,idx].min()
#                               for idx in xrange(input_frames.shape[2])))
#        reference_frame = input_frames[:,:,ranges.argmax()]


########## Align frames to *centre* frame

aligned_frames = align(input_frames, reference_frame)
mean_aligned_frame = np.mean(aligned_frames, axis=2)


######### Register frames

registration_reference = mean_aligned_frame
#registration_reference = reference_frame

registered_frames = register(aligned_frames, registration_reference)

########## Transform registered frames
lowpasses, highpasses = transform_frames(registered_frames)

########## Compute mean lowpass image
lowpass_mean = np.mean(lowpasses, axis=2)

########## Get mean direction for each subband
phases = []
for level_sb in highpasses:
    
    # Calculate mean direction by adding all subbands together and normalising
    sum_ = np.sum(level_sb, axis=3)
    sum_mag = np.abs(sum_)
    sum_ /= np.where(sum_mag != 0, sum_mag, 1)
    phases.append(sum_)

########## Compute mean, maximum and maximum-of-inliers magnitudes
mean_mags, max_mags, max_inlier_mags = [], [], []
for level_sb in highpasses:
    mags = np.abs(level_sb)

    mean_mags.append(np.mean(mags, axis=3))
    max_mags.append(np.max(mags, axis=3))

    thresh = 2*np.repeat(np.median(mags, axis=3)[:,:,:,np.newaxis], level_sb.shape[3], axis=3)
    outlier_suppressed = np.where(mags < thresh, mags, 0)
    max_inlier_mags.append(np.max(outlier_suppressed, axis=3))

########## Reconstruct frames
mean_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(mean_mags, phases)))
#save_image(imprefix + 'fused-mean-dtcwt', mean_recon)

max_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(max_mags, phases)))
#save_image(imprefix + 'fused-max-dtcwt', max_recon)

max_inlier_recon = reconstruct(lowpass_mean, tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases)))
#save_image(imprefix + 'fused-max-inlier-dtcwt', max_inlier_recon)

max_inlier_shrink_recon = reconstruct(lowpass_mean,
        shrink_coeffs(tuple(mag*phase for mag, phase in zip(max_inlier_mags, phases))))
#save_image(imprefix + 'fused-max-inlier-shrink-dtcwt', max_inlier_shrink_recon)

plt.figure(1)
plt.title('input image(1)'), plt.xticks([]), plt.yticks([])
plt.imshow(input_frames[:,:,0],'gray')
 
plt.figure(2)
plt.title('fused image_max_inlier_recon'), plt.xticks([]), plt.yticks([])
plt.imshow(max_inlier_recon,'gray')

plt.figure(3)
plt.title('fused image_max_recon'), plt.xticks([]), plt.yticks([])
plt.imshow(max_recon,'gray')

