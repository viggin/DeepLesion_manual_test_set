"""
Utilities to compute the accuracy for lesion detection.
"""
import numpy as np
from scipy import interpolate


def sens_at_FP(sens, fp_per_img, FP_values):
    """
    compute the sensitivity at certain average FPs per image.
    :param sens: Sensitivity values on an FROC curve, a numpy array with length n.
    :param fp_per_img: Average FP values on an FROC curve, a numpy array with length n.
    :param FP_values: The FP values you want to compute sensitivity on, a numpy array with length m.
    :return: The sensitivity values, a numpy array with length m.
    """
    avgFP_out_left = [a for a in FP_values if a < fp_per_img[0]]
    avgFP_in = [a for a in FP_values if fp_per_img[0] <= a <= fp_per_img[-1]]
    avgFP_out = [a for a in FP_values if a > fp_per_img[-1]]
    f = interpolate.interp1d(fp_per_img, sens)
    sens_out = np.hstack([np.zeros((len(avgFP_out_left, ))), f(np.array(avgFP_in)), np.ones((len(avgFP_out, )))*sens[-1]])
    return sens_out


def FROC(boxes_all, gts_all, dim='3D', overlap_metric='IOU', overlap_th=.3):
    """
    Compute the Free ROC curve of single class object detection.
    :param boxes_all: A list of numpy arrays. The i'th array is the predicted boxes in the i'th image. Its size is
     n_i x d, n_i the number of the boxes in the image, d = 5 for 2D boxes (x1, y1, x2, y2, score), and 7 for 3D boxes
     (x1, y1, z1, x2, y2, z2, score).
     If there is no box in one image, the array should have size 0 x d.
    :param gts_all: A list of numpy arrays. The i'th array is the ground-truth boxes in the i'th image. Its size is
     g_i x D, g_i the number of the boxes in the image, d = 4 for 2D boxes (x1, y1, x2, y2), and 6 for 3D boxes
     (x1, y1, z1, x2, y2, z2).
    If there is no gt in one image, the array should have size 0 x d.
    If you need to compute a stratified FROC (split the gt according to box size, type, etc.), you can call this
     function multiple times with different gts_all as inputs.
    :param dim: One of the 3 strings: 2D, 3D, P3D. P3D means detections are in 3D but gts are in 2D. If a gt is on one
     slice of the box, we compute the 2D overlap to decide if the box hits the gt. Note that gt is only annotated on one
     slice, but its format should still in 3D (x1, y1, z, x2, y2, z).
    :param overlap_metric: One of the 4 strings: IOU, IOBB, IOGT, IOBBGT. IOBBGT means the box is regarded as hit if
    either IOBB or IOGT >= overlap_th, which is a looser metric.
    :param overlap_th: If overlap_metric >= overlap_th, a box is regarded as hit.
    :return: sens, fp_per_img, thresholds: 3 numpy arrays with the same length m, which is the total number of boxes in
     all images. If we set the score threshold = thresholds[i], the sensitivity (recall) will be sens[i] with fp_per_img[i]
     false positives.
    """
    assert len(boxes_all) == len(gts_all), "There should be same numbers of images"
    if dim in ('3D', 'P3D'):
        assert boxes_all[0].shape[1] == 7
        assert gts_all[0].shape[1] == 6
    elif dim == '2D':
        assert boxes_all[0].shape[1] == 5
        assert gts_all[0].shape[1] == 4

    nImg = len(boxes_all)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :-1]
    img_idxs = img_idxs[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []

    for i in range(len(boxes_cat)):
        overlap = IOX(boxes_cat[i, :], gts_all[img_idxs[i]], dim, overlap_metric)
        if len(overlap) == 0 or np.any(np.isnan(overlap)) or (overlap.max() < overlap_th):
            nMiss += 1
        else:
            for j in range(len(overlap)):
                if overlap[j] >= overlap_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)

    # nGt = len(np.vstack([gts for gts in gts_all if not np.any(np.isnan(gts))]))
    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg

    return sens, fp_per_img, scores


def IOX(box1, gts, dim='3D', metric='IOU'):
    """
    Compute the IOU, IOBB, IOGT, or IOBBGT between a box and a list of boxes. We assume the box is a predicted bounding-box and
    the box list is a list of ground-truths of the same image. IOU is intersection over union; IOBB is intersection over
    predicted bounding-box; IOGT is intersection over ground-truths; IOBBGT is max(IOBB, IOGT)
    :param box1: A numpy array of size d. d = 4 for 2D box (x1, y1, x2, y2), and 6 for 3D box (x1, y1, z1, x2, y2, z2).
    :param gts: A numpy array of size n x d. n is the number of gt boxes.
    :param dim: One of the 3 strings: 2D, 3D, P3D. P3D means detections are in 3D but gts are in 2D. If a gt is on one
     slice of the box, we compute the 2D overlap to decide if the box hits the gt. Note that gt is only annotated on one
     slice, but its format should still in 3D (x1, y1, z, x2, y2, z).
    :param metric: One of the 4 strings: IOU, IOBB, IOGT, IOBBGT.
    :return: A numpy array with size n.
    """
    if dim == '3D':
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        izmin = np.maximum(gts[:, 2], box1[2])
        ixmax = np.minimum(gts[:, 3], box1[3])
        iymax = np.minimum(gts[:, 4], box1[4])
        izmax = np.minimum(gts[:, 5], box1[5])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        id = np.maximum(izmax - izmin + 1., 0.)
        inters = iw * ih * id
        union = ((box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.) +
               (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) * (gts[:, 5] - gts[:, 2] + 1.) - inters)
        box_size = (box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.)
        gt_sizes = (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) * (gts[:, 5] - gts[:, 2] + 1.)

    elif dim == 'P3D':
        assert np.all(gts[:, 2] == gts[:, 5])
        z_inters = (box1[2] <= gts[:, 2]) & (gts[:, 2] <= box1[5])
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        ixmax = np.minimum(gts[:, 3], box1[3])
        iymax = np.minimum(gts[:, 4], box1[4])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih * z_inters
        union = ((box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) +
               (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) - inters)
        box_size = (box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.)
        gt_sizes = (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.)

    elif dim == '2D':
        ixmin = np.maximum(gts[:, 0], box1[0])
        iymin = np.maximum(gts[:, 1], box1[1])
        ixmax = np.minimum(gts[:, 2], box1[2])
        iymax = np.minimum(gts[:, 3], box1[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        union = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
               (gts[:, 2] - gts[:, 0] + 1.) * (gts[:, 3] - gts[:, 1] + 1.) - inters)
        box_size = (box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.)
        gt_sizes = (gts[:, 2] - gts[:, 0] + 1.) * (gts[:, 3] - gts[:, 1] + 1.)

    if metric == 'IOU':
        return inters / union
    elif metric == 'IOBB':
        return inters / box_size
    elif metric == 'IOGT':
        return inters / gt_sizes
    elif metric == 'IOBBGT':
        return np.maximum(inters / box_size, inters / gt_sizes)
