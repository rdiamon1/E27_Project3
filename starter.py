#!/usr/bin/env python

import cv2
import numpy
import sys
import os

f = 600
u0 = 320
v0 = 240
b = 0.05

K = numpy.matrix('600 0 320; 0 600 240; 0 0 1')
print 'K = ', K

Kinv = numpy.linalg.inv(K)

print 'Kinv = ', Kinv

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

cam_image[0,1] = 255 ## REMOVE THIS!!!

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)
width = cam_image.shape[1]
height = cam_image.shape[0]
print 'width = ',width,' height = ',height

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

n = disparity.shape[0] * disparity.shape[1]

print disparity.shape
print n

# vectorized version
w = 15
h = 10
print cam_image[0:h,0:w]
xcoords = range(width)
ycoords = range(height)
X,Y = numpy.meshgrid(xcoords, ycoords)
print X[0:h,0:w]
print Y[0:h,0:w]
#mask = numpy.equal(cam_image[X,Y],255)
#mask = numpy.logical_and(cam_image[X,Y]==255 and True)
mask = numpy.equal(cam_image,255)
print mask[0:h,0:w]
print cam_image[mask].shape
print cam_image[mask][:50]

#xpts = numpy.logical_and(X,mask)
xpts = X[mask]
ypts = Y[mask]
zpts = numpy.ones(xpts.shape)
pts = numpy.vstack((xpts,ypts,zpts))
print 'now xpts'
print xpts.shape
print ypts.shape
print zpts.shape
print pts.shape


qvals = numpy.dot(Kinv, cam_image[mask])


# iterative version
qarr = numpy.zeros((n, 3))
qindx = 0
for row in range(height):
    for col in range(width):
        if cam_image[row][col] == 255:
            qarr[qindx] = numpy.dot(Kinv, [row, col, 1])
            qindx += 1
print 'qindx = ', qindx
qarr = qarr[:qindx-1]



Z = (b * f) / disparity
Zmax = 8

# qarr =

print cam_image.shape

# Pop up the disparity image.
cv2.imshow('Disparity', disparity/disparity.max())
while cv2.waitKey(5) < 0: pass
