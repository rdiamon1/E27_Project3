"""
starter_clean.py

Stater code:   Matt Zucker
Modifications: Julie Harris and Rachel Diamond
Date:          April 2016

This program takes in two .png images (one with projected points and the other
with the new locations of those points after they are distorted by a 3D object)

It creates a nx3 array of of XYZ data to be used as input to the PointCloudApp
module, where n is the useable number of pixels from the disparity image.
"""

import cv2
import numpy
import sys
import os

# name of the file for the resulting array
returnfilename = 'starterXYZ'

# define variables for K matrix
f = 600
u0 = 320
v0 = 240
b = 0.05

# define K matrix
K = numpy.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])

# define K inverse by taking the inverse of K
Kinv = numpy.linalg.inv(K)

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)


# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)
width = cam_image.shape[1]
height = cam_image.shape[0]

# Set up parameters for stereo matching (see OpenCV docs at
# http://goo.gl/U5iW51 for details).
min_disparity = 0
max_disparity = 16
window_size = 11
param_P1 = 0
param_P2 = 20000

# Create a stereo matcher object
matcher = cv2.StereoSGBM_create(min_disparity,
                                max_disparity,
                                window_size,
                                param_P1,
                                param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparities.
disparity = matcher.compute(cam_image, proj_image) / 16.0

# vectorized version
ucoords = range(width)
vcoords = range(height)
U,V = numpy.meshgrid(ucoords, vcoords)

# set maximum Z value (distance from camera) as defined in project description.
Zmax = 8 # this is 8 meters

# set a minimum disparity value using equation given in project description.
disp_low = (b*f)/Zmax

# define a set of points where the disparity is greater
# than the above defined minimum value.
mask = (disparity > disp_low)

# define the points in U and V that have the correct disparity
upts = U[mask]
vpts = V[mask]
onepts = numpy.ones(upts.shape)

# stack the u points, v points, and 1 points to create the q array
pts = numpy.vstack((upts,vpts,onepts))

# define disparity points by selecting the points in
# disparity that are already selected by the mask
disp_pts = disparity[mask]

# define an array of points by multiplying K inverse by array of q points
Parr = numpy.dot(Kinv, pts)

# define an array of Z points using the previously
# defined K-array variables and disparity points
Z = (b * f) / disp_pts

# Pop up the disparity image.
cv2.imshow('Disparity', disparity/disparity.max())
while cv2.waitKey(5) < 0: pass

ParrScaled = Parr * Z
ParrScaled = numpy.transpose(ParrScaled)

# Save the properly scaled array as starterXYZ.npy
numpy.save(returnfilename,ParrScaled)
