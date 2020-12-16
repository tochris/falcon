import numpy as np
from scipy.special import softmax


def calculate_expected_calibration_error(
    logits, labels,
    bins_calibration):
    """Calculate ECE from logits and labels.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
        bins_calibration: list of floats, limits of calibration bins
    Returns:
        ECE: float
    """

    def probability_accuracy_binned(
            logits, labels,
            bins_calibration = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        """Calculate binned probability and accuracy from logits and labels.
        Args:
            logits: array, (samples, logits)
            labels: array, (samples, one hot labels)
            bins_calibration: list of floats, limits of calibration bins
        Returns:
            binned probability, binned accuracy arrays, maximum probability array
        """
        pred = np.argmax(logits, axis = 1)
        truth = np.argmax(labels, axis = 1)
        match = np.equal(pred, truth)
        probs = softmax(logits, axis = 1)
        probs = np.clip(probs, 1.17e-37, 3.40e37)
        max_prob = np.ndarray.max(probs, axis = 1)

        binned_probs = np.zeros(np.shape(bins_calibration)[0]+1)
        binned_accs = np.zeros(np.shape(bins_calibration)[0]+1)
        #Calculate to which bin an array element belongs to
        binplace = np.digitize(max_prob, bins_calibration, right = True)
        for bin_num in range(0, np.shape(bins_calibration)[0]+1):
            if bin_num in binplace:
                to_replace = np.where(binplace == bin_num)
                binned_probs[bin_num] = np.mean(np.array(max_prob)[to_replace])
                binned_accs[bin_num] = np.mean(np.array(match)[to_replace])
        return binned_probs, binned_accs, max_prob

    binned_probability_vec, binned_accuracy_vec, max_prob = probability_accuracy_binned(
        logits, labels, bins_calibration
    )
    frequency_in_bins_list = np.zeros(np.shape(bins_calibration)[0]+1)
    #Calculate to which bin an array element belongs to
    binplace = np.digitize(max_prob, bins_calibration, right=True)
    for bin_num in range(0,np.shape(bins_calibration)[0]+1):
        if bin_num in binplace:
            frequency_in_bins_list[bin_num]=np.shape(
                np.array(max_prob)[np.where(binplace == bin_num)])[0]
    ece=np.sum((frequency_in_bins_list/np.sum(frequency_in_bins_list))* \
        np.absolute(np.array(binned_accuracy_vec) - np.array(binned_probability_vec)))
    return ece
