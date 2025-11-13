"""
Adapted from the following references:
[1] https://github.com/JunMa11/NeurIPS-CellSeg/blob/main/baseline/compute_metric.py
[2] https://github.com/stardist/stardist/blob/master/stardist/matching.py

"""

import numpy as np
from skimage import segmentation
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix 
from numba import jit



__all__ = ["evaluate_f1_score_cellseg", "evaluate_f1_score"]


def evaluate_metrics_cellseg(pred_mask, gt_mask, threshold=0.5):
    """
    Computes IoU, Precision, Recall, and F1-score for 3D cell segmentation masks.

    Parameters:
        gt_mask (numpy.ndarray): 3D ground truth mask (binary or labels)
        pred_mask (numpy.ndarray): 3D predicted mask (float or binary)
        threshold (float): Threshold to binarize the predicted mask (if not already binary)

    Returns:
        tuple: (iou, precision, recall, f1_score)
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Shape mismatch: ground truth and predicted masks must have the same shape.")


    # Ensure masks are binary
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    iou = intersection / union if union != 0 else 1.0
    precision = intersection / pred_sum if pred_sum != 0 else 1.0
    recall = intersection / gt_sum if gt_sum != 0 else 1.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return iou, precision, recall, f1_score


def _intersection_over_union(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs.

    Parameters:
        masks_true (np.ndarray, int): Ground truth masks, where 0=NO masks; 1,2... are mask labels.
        masks_pred (np.ndarray, int): Predicted masks, where 0=NO masks; 1,2... are mask labels.

    Returns:
        iou (np.ndarray, float): Matrix of IOU pairs of size [x.max()+1, y.max()+1].

    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 
    """
    if masks_true.size != masks_pred.size:
        raise ValueError("masks_true.size != masks_pred.size")
    mask = (masks_true > 0) | (masks_pred > 0)
    overlap = csr_matrix(
        (np.ones(mask.sum(), int),
        (masks_true[mask].ravel(), masks_pred[mask].ravel())),
        shape=(masks_true.max() + 1, masks_pred.max() + 1)
    )
    overlap = overlap.toarray()
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def dice_per_annotated_cell_3d(pred, gt):
    """
    Compute Dice for each annotated GT cell in 3D instance segmentation.
    pred, gt: 3D numpy arrays (Z, Y, X)
    """
    dice_scores = []
    
    for cell_id in np.unique(gt):
        if cell_id == 0:
            continue  # skip background

        gt_mask = (gt == cell_id).astype(np.uint8)
        # Find which predicted labels overlap with this GT cell
        overlap_labels, counts = np.unique(pred[gt_mask > 0], return_counts=True)
        overlap_labels = overlap_labels[overlap_labels != 0]  # remove background

        if len(overlap_labels) == 0:
            dice_scores.append(0.0)  # missed cell
            continue

        # Take predicted instance with the maximum overlap
        best_pred_label = overlap_labels[np.argmax(counts)]
        pred_mask = (pred == best_pred_label).astype(np.uint8)

        intersection = np.sum(gt_mask * pred_mask)
        dice = 2 * intersection / (np.sum(gt_mask) + np.sum(pred_mask) + 1e-8)
        dice_scores.append(dice)

    return np.mean(dice_scores), dice_scores


def _true_positive(iou, th):
    """Calculate the true positive at threshold th.

    Args:
        iou (float, np.ndarray): Array of IOU pairs.
        th (float): Threshold on IOU for positive label.

    Returns:
        tp (float): Number of true positives at threshold.

    How it works:
        (1) Find minimum number of masks.
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...).
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs from these pairings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

import numpy as np
from scipy.optimize import linear_sum_assignment

def fast_iou_matrix(masks_true, masks_pred):
    labels_true = masks_true.ravel()
    labels_pred = masks_pred.ravel()
    valid = (labels_true > 0) | (labels_pred > 0)
    labels_true = labels_true[valid]
    labels_pred = labels_pred[valid]

    # compute overlap using bincount (faster than CSR)
    n_true = masks_true.max() + 1
    n_pred = masks_pred.max() + 1
    overlap = np.bincount(
        labels_true * n_pred + labels_pred,
        minlength=n_true * n_pred
    ).reshape(n_true, n_pred)

    # compute IoU
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    union = n_pixels_pred + n_pixels_true - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap, float), where=union > 0)
    return iou[1:, 1:]  # remove background


def true_positives(iou, th):
    if iou.size == 0:
        return 0
    costs = -(iou >= th).astype(float) - iou / (2 * min(iou.shape))
    true_ind, pred_ind = linear_sum_assignment(costs)
    return np.sum(iou[true_ind, pred_ind] >= th)


def average_precision_final(masks_true, masks_pred, threshold=0.5):
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]

    thresholds = np.atleast_1d(threshold)
    ap = np.zeros((len(masks_true), len(thresholds)), float)

    for n, (gt, pred) in enumerate(zip(masks_true, masks_pred)):
        iou = fast_iou_matrix(gt, pred)
        n_true, n_pred = gt.max(), pred.max()
        for k, th in enumerate(thresholds):
            tp = true_positives(iou, th)
            fp = n_pred - tp
            fn = n_true - tp
            ap[n, k] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return ap.squeeze()


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ 
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)): 
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)): 
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]): 
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]): 
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]): 
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]): 
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap

def compute_CTC_SEG(gt_mask, pred_mask):
    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels != 0]

    per_object_J = {}

    for gt_id in gt_labels:
        R = gt_mask == gt_id
        J_val = 0.0

        for pred_id in pred_labels:
            S = pred_mask == pred_id
            inter = np.logical_and(R, S).sum()
            if inter <= 0.5 * R.sum():
                continue  # not enough overlap to count as a match

            union = np.logical_or(R, S).sum()
            J_val = inter / union
            break  # since no other prediction can pass the >0.5 test

        per_object_J[int(gt_id)] = J_val

    seg_score = np.mean(list(per_object_J.values())) if per_object_J else 0.0
    return seg_score, per_object_J

from scipy.sparse import coo_matrix

def compute_AP_fast(gt_mask, pred_mask, iou_threshold=0.5):
    """
    Fast AP at given IoU threshold for 2D/3D instance masks, works with non-contiguous labels.
    """
    gt_mask = gt_mask.astype(np.int64)
    pred_mask = pred_mask.astype(np.int64)

    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels != 0]

    if len(gt_labels) == 0 and len(pred_labels) == 0:
        return 1.0
    if len(gt_labels) == 0 or len(pred_labels) == 0:
        return 0.0

    # Flatten
    gt_flat = gt_mask.ravel()
    pred_flat = pred_mask.ravel()

    # Sparse co-occurrence matrix
    data = np.ones_like(gt_flat, dtype=np.int64)
    overlap = coo_matrix(
        (data, (gt_flat, pred_flat)),
        shape=(gt_mask.max() + 1, pred_mask.max() + 1)
    ).tocsc()

    # Compute sizes
    gt_sizes = np.bincount(gt_flat)[0 : gt_mask.max() + 1]
    pred_sizes = np.bincount(pred_flat)[0 : pred_mask.max() + 1]

    # Build IoU matrix
    iou_matrix = np.zeros((len(gt_labels), len(pred_labels)), dtype=np.float32)
    for i, gt_id in enumerate(gt_labels):
        row = overlap.getrow(gt_id)
        pred_ids = row.indices[row.indices != 0]  # remove background
        intersections = row.data[row.indices != 0]
        if intersections.size > 0:
            unions = gt_sizes[gt_id] + pred_sizes[pred_ids] - intersections
            iou_matrix[i, [np.where(pred_labels == pid)[0][0] for pid in pred_ids]] = intersections / unions

    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    tp = 0

    for i, _ in enumerate(gt_labels):
        j_best = np.argmax(iou_matrix[i])
        if iou_matrix[i, j_best] >= iou_threshold:
            tp += 1
            matched_gt.add(i)
            matched_pred.add(j_best)

    fp = len(pred_labels) - len(matched_pred)
    fn = len(gt_labels) - len(matched_gt)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    ap = 2 * precision * recall / (precision + recall + 1e-8) if tp > 0 else 0.0

    return ap

def compute_CTC_SEG_fast(gt_mask, pred_mask):
    """
    Fast SEG computation for large 3D volumes.
    Computes overlaps in one pass using sparse matrices.

    Parameters
    ----------
    gt_mask : np.ndarray, int
        Ground truth labeled mask (3D).
    pred_mask : np.ndarray, int
        Predicted labeled mask (3D).

    Returns
    -------
    seg_score : float
        Mean Jaccard index (SEG measure).
    per_object_J : dict
        Mapping: GT object ID -> Jaccard index.
    """
    gt_mask = gt_mask.astype(np.int64)
    pred_mask = pred_mask.astype(np.int64)

    gt_labels = np.unique(gt_mask)
    gt_labels = gt_labels[gt_labels != 0]
    pred_labels = np.unique(pred_mask)
    pred_labels = pred_labels[pred_labels != 0]

    # Flatten the arrays
    gt_flat = gt_mask.ravel()
    pred_flat = pred_mask.ravel()

    # Build sparse co-occurrence matrix of overlaps
    data = np.ones_like(gt_flat, dtype=np.int64)
    overlap = coo_matrix(
        (data, (gt_flat, pred_flat)),
        shape=(gt_mask.max() + 1, pred_mask.max() + 1)
    ).tocsc()

    # Precompute voxel counts per label
    gt_sizes = np.bincount(gt_flat)[0 : gt_mask.max() + 1]
    pred_sizes = np.bincount(pred_flat)[0 : pred_mask.max() + 1]

    per_object_J = {}

    for gt_id in gt_labels:
        gt_size = gt_sizes[gt_id]
        row = overlap.getrow(gt_id)

        # Get nonzero overlaps (pred IDs and counts)
        pred_ids = row.indices
        intersections = row.data

        if intersections.size == 0:
            per_object_J[gt_id] = 0.0
            continue

        # Apply >0.5*|R| criterion
        valid = intersections > 0.5 * gt_size
        if not np.any(valid):
            per_object_J[gt_id] = 0.0
            continue

        inter = intersections[valid][0]
        pred_id = pred_ids[valid][0]
        union = gt_size + pred_sizes[pred_id] - inter
        per_object_J[gt_id] = inter / union

    seg_score = np.mean(list(per_object_J.values())) if per_object_J else 0.0
    return seg_score, per_object_J


def evaluate_f1_score_cellseg(masks_true, masks_pred, threshold=0.5):
    """
    Get confusion elements for cell segmentation results.
    Boundary pixels are not considered during evaluation.
    """

    if np.prod(masks_true.shape) < (5000 * 5000):
        masks_true = _remove_boundary_cells(masks_true.astype(np.int32))
        masks_pred = _remove_boundary_cells(masks_pred.astype(np.int32))

        tp, fp, fn = get_confusion(masks_true, masks_pred, threshold)

    # Compute by Patch-based way for large images
    else:
        H, W = masks_true.shape
        roi_size = 2000

        # Get patch grid by roi_size
        if H % roi_size != 0:
            n_H = H // roi_size + 1
            new_H = roi_size * n_H
        else:
            n_H = H // roi_size
            new_H = H

        if W % roi_size != 0:
            n_W = W // roi_size + 1
            new_W = roi_size * n_W
        else:
            n_W = W // roi_size
            new_W = W

        # Allocate values on the grid
        gt_pad = np.zeros((new_H, new_W), dtype=masks_true.dtype)
        pred_pad = np.zeros((new_H, new_W), dtype=masks_true.dtype)
        gt_pad[:H, :W] = masks_true
        pred_pad[:H, :W] = masks_pred

        tp, fp, fn = 0, 0, 0

        # Calculate confusion elements for each patch
        for i in range(n_H):
            for j in range(n_W):
                gt_roi = _remove_boundary_cells(
                    gt_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                )
                pred_roi = _remove_boundary_cells(
                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                )
                tp_i, fp_i, fn_i = get_confusion(gt_roi, pred_roi, threshold)
                tp += tp_i
                fp += fp_i
                fn += fn_i

    # Calculate f1 score
    precision, recall, f1_score = evaluate_f1_score(tp, fp, fn)

    return precision, recall, f1_score


def evaluate_f1_score(tp, fp, fn):
    """Evaluate F1-score for the given confusion elements"""

    # Do not Compute on trivial results
    if tp == 0:
        precision, recall, f1_score = 0, 0, 0

    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def _remove_boundary_cells(mask):
    """Remove cells on the boundary from a 2D or 3D mask."""
    
    if mask.ndim == 2:
        W, H = mask.shape
        bd = np.ones((W, H), dtype=bool)
        bd[2:W-2, 2:H-2] = 0
        bd_cells = np.unique(mask * bd)
        
    elif mask.ndim == 3:
        D, W, H = mask.shape
        bd = np.zeros((D, W, H), dtype=bool)
        
        # Set boundary faces to True
        bd[0, :, :] = True
        bd[-1, :, :] = True
        bd[:, 0, :] = True
        bd[:, -1, :] = True
        bd[:, :, 0] = True
        bd[:, :, -1] = True
        
        bd_cells = np.unique(mask[bd])
        
    else:
        raise ValueError("Only 2D or 3D masks are supported.")

    # Remove boundary cells
    for i in bd_cells[bd_cells != 0]:
        mask[mask == i] = 0

    # Relabel sequentially
    new_label, _, _ = segmentation.relabel_sequential(mask)

    return new_label


def get_confusion(masks_true, masks_pred, threshold=0.5):
    """Calculate confusion matrix elements: (TP, FP, FN)"""
    num_gt_instances = np.max(masks_true)
    num_pred_instances = np.max(masks_pred)

    if num_pred_instances == 0:
        print("No segmentation results!")
        tp, fp, fn = 0, 0, 0

    else:
        # Calculate IoU and exclude background label (0)
        iou = _get_iou(masks_true, masks_pred)
        iou = iou[1:, 1:]

        # Calculate true positives
        tp = _get_true_positive(iou, threshold)
        fp = num_pred_instances - tp
        fn = num_gt_instances - tp

    return tp, fp, fn


def _get_true_positive(iou, threshold=0.5):
    """Get true positive (TP) pixels at the given threshold"""

    # Number of instances to be matched
    num_matched = min(iou.shape[0], iou.shape[1])

    # Find optimal matching by using IoU as tie-breaker
    costs = -(iou >= threshold).astype(float) - iou / (2 * num_matched)
    matched_gt_label, matched_pred_label = linear_sum_assignment(costs)

    # Consider as the same instance only if the IoU is above the threshold
    match_ok = iou[matched_gt_label, matched_pred_label] >= threshold
    tp = match_ok.sum()

    return tp


def _get_iou(masks_true, masks_pred):
    """Get the iou between masks_true and masks_pred"""

    # Get overlap matrix (GT Instances Num, Pred Instance Num)
    overlap = _label_overlap(masks_true, masks_pred)

    # Predicted instance pixels
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)

    # GT instance pixels
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)

    # Calculate intersection of union (IoU)
    union = n_pixels_pred + n_pixels_true - overlap
    iou = overlap / union

    # Ensure numerical values
    iou[np.isnan(iou)] = 0.0

    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """Get pixel overlaps between two masks

    Parameters
    ------------
    x, y (np array; dtype int): 0=NO masks; 1,2... are mask labels

    Returns
    ------------
    overlap (np array; dtype int): Overlaps of size [x.max()+1, y.max()+1]
    """

    # Make as 1D array
    x, y = x.ravel(), y.ravel()

    # Preallocate a Contact Map matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # Calculate the number of shared pixels for each label
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1

    return overlap
