"""
A simple demo to load 2D 16-bit slices from DeepLesion and save to 3D nifti volumes.
The nifti volumes can be viewed in software such as 3D slicer and ITK-SNAP.
This code is slightly diffferent from the one in the official dataset.
"""


import numpy as np
import nibabel as nib
import os
import cv2
import csv
import SimpleITK as sitk


dir_in = 'Images_png'
dir_out = 'Images_nifti'
out_fmt = '%s_%03d-%03d.nii.gz'  # format of the nifti file name to output
info_fn = 'DL_info.csv'  # file name of the information file


def slices2nifti(ims, fn_out, spacing):
    """save 2D slices to 3D nifti file considering the spacing"""
    if len(ims) < 300:  # cv2.merge does not support too many channels
        V = cv2.merge(ims)
    else:
        V = np.empty((ims[0].shape[0], ims[0].shape[1], len(ims)))
        for i in range(len(ims)):
            V[:, :, i] = ims[i]
    V = V.transpose((2, 0, 1))
    V = V[::-1]  # RAI, to be consistent with other datasets
    
    path_out = os.path.join(dir_out, fn_out)
    V = sitk.GetImageFromArray(V)
    V.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    V.SetSpacing(spacing)
    sitk.WriteImage(V, path_out)

    print(fn_out, 'saved')


def load_slices(dir, slice_idxs):
    """load slices from 16-bit png files"""
    slice_idxs = np.array(slice_idxs)
    assert np.all(slice_idxs[1:] - slice_idxs[:-1] == 1)
    ims = []
    for slice_idx in slice_idxs:
        fn = '%03d.png' % slice_idx
        path = os.path.join(dir_in, dir, fn)
        im = cv2.imread(path, -1)  # -1 is needed for 16-bit image
        assert im is not None, 'error reading %s' % path
        print('read', path)

        # the 16-bit png file has a intensity bias of 32768
        ims.append((im.astype(np.int32) - 32768).astype(np.int16))
    return ims


def read_DL_info():
    """read spacings and image indices in DeepLesion"""
    spacings = []
    idxs = []
    with open(info_fn, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rownum = 0
        for row in reader:
            if rownum == 0:
                header = row
                rownum += 1
            else:
                idxs.append([int(d) for d in row[1:4]])
                spacings.append([float(d) for d in row[12].split(',')])

    idxs = np.array(idxs)
    spacings = np.array(spacings)
    return idxs, spacings


if __name__ == '__main__':
    idxs, spacings = read_DL_info()
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    img_dirs = os.listdir(dir_in)
    img_dirs.sort()
    for dir1 in img_dirs:
        # find the image info according to the folder's name
        idxs1 = np.array([int(d) for d in dir1.split('_')])
        i1 = np.where(np.all(idxs == idxs1, axis=1))[0]
        spacings1 = spacings[i1[0]]

        fns = os.listdir(os.path.join(dir_in, dir1))
        slices = [int(d[:-4]) for d in fns if d.endswith('.png')]
        slices.sort()

        # Each folder contains png slices from one series (volume)
        # There may be several sub-volumes in each volume depending on the key slices
        # We group the slices into sub-volumes according to continuity of the slice indices
        groups = []
        for slice_idx in slices:
            if len(groups) != 0 and slice_idx == groups[-1][-1]+1:
                groups[-1].append(slice_idx)
            else:
                groups.append([slice_idx])

        for group in groups:
            # group contains slices indices of a sub-volume
            ims = load_slices(dir1, group)
            fn_out = out_fmt % (dir1, group[0], group[-1])
            slices2nifti(ims, fn_out, spacings1)
