"""
Sample evaluation code of the paper
“Learning from Multiple Datasets with Heterogeneous and Partial Labels for Universal Lesion Detection in CT”
in IEEE Trans. Med. Imaging, 2020
"""
import pickle
import numpy as np

from general_detection_evaluation import sens_at_FP, FROC


IOU_METRIC = 'IOBB'
IOU_TH_3D = .3


def evaluate(gts, boxes_pred):
    froc_fp = [2 ** p for p in range(-3, 11)]  # average FP=0.125, ..., 1024 per sub-volume
    fns = list(gts.keys())
    all_boxes = [boxes_pred[fn] for fn in fns]
    all_gts = [gts[fn] for fn in fns]
    sens, fps, score_ths = FROC(all_boxes, all_gts, '3D', IOU_METRIC, IOU_TH_3D)
    det_res = sens_at_FP(sens, fps, froc_fp)
    det_res = np.hstack((det_res, det_res[:7].mean()))

    np.set_printoptions(precision=4)
    print(f'Sensitivity at FP={froc_fp} per sub-volume:\n{det_res}')
    print(f'Average sensitivity at FP=0.125:8 per sub-volume:', np.mean(det_res[:7]))


def main():
    # gts is a dict, gts[fn] = B, where fn is the name of the sub-volume.
    # B is an n-by-6 numpy array, n is the number of 3D boxes in the sub-volume.
    # It is possible that a sub-volume has zero boxes.
    # Each row of B is a 3D box in the format of [x1, y1, z1, x2, y2, z2],
    # where z1 and z2 are the slice indices of the whole volume (not the slice indices of this sub-volume).
    # The name format of a sub-volume is "{volume_name}_{top slice index}-{bottom slice index}", see DL_save_nifti.py

    save_fn = 'DeepLesion_manual_1K_test_release.pkl'
    gts = pickle.load(open(save_fn, 'rb'))['test']  # or ['val']

    # modify this line to put your predicted 3D boxes as the same format as gts, except that
    # boxes_pred[fn] is an n-by-7 numpy array with the last column being the confidence score
    boxes_pred = {k: np.hstack((v, np.ones((len(v), 1)))) for k, v in gts.items()}
    evaluate(gts, boxes_pred)


if __name__ == '__main__':
    main()
