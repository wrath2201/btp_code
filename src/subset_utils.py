import numpy as np
import src.metrics_perclass
import src.pqmodel
import src.dasnet
import src.dualpq

# 1-based indexing as in original code
SUBSETS = {
    "easy17": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 25, 27],
    "easy21": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 24, 25, 26, 27],
    "easy25": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27]
}

def apply_subset(subset_name):
    if not subset_name:
        return None
    
    if subset_name not in SUBSETS:
        raise ValueError(f"Unknown subset: {subset_name}")
        
    subset = SUBSETS[subset_name]
    K = len(subset)
    
    # Monkey-patch N_CLASSES globally
    src.metrics_perclass.N_CLASSES = K
    src.pqmodel.N_CLASSES = K
    src.dasnet.N_CLASSES = K
    src.dualpq.N_CLASSES = K
    
    return subset

def filter_dataset(W, X, y, group, snr, subset):
    """
    Filters the dataset arrays to only include rows where y is in `subset`.
    Then remaps `y` to contiguous integers 0 to K-1 (or 1 to K if original was 1-based).
    """
    y = np.asarray(y)
    is_1_based = (y.min() >= 1)
    
    if is_1_based:
        subset_y = np.array(subset)
    else:
        subset_y = np.array(subset) - 1
        
    mask = np.isin(y, subset_y)
    
    W_f = W[mask] if W is not None else None
    X_f = X[mask] if X is not None else None
    y_f = y[mask]
    group_f = group[mask] if group is not None else None
    snr_f = snr[mask] if snr is not None else None
    
    # Remap labels
    y_mapped = np.zeros_like(y_f)
    for i, orig_class in enumerate(subset_y):
        y_mapped[y_f == orig_class] = (i + 1) if is_1_based else i
        
    return W_f, X_f, y_mapped, group_f, snr_f, mask

def remap_predictions_to_original(yp_mapped, subset, is_1_based=True):
    """
    Maps 0..K-1 (or 1..K) predictions back to the original 1..29 (or 0..28) labels.
    """
    if is_1_based:
        subset_y = np.array(subset)
        yp_mapped = np.asarray(yp_mapped) - 1 # convert to 0-based for indexing
    else:
        subset_y = np.array(subset) - 1
        yp_mapped = np.asarray(yp_mapped)
        
    return subset_y[yp_mapped]

