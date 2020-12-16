import numpy as np
import tensorflow as tf
import source.utilsevaluation.calibration as calib
from scipy.special import softmax
from sklearn.metrics import brier_score_loss, log_loss


def accuracy(logits, labels, **params):
    """Calculate accuracy from logits and labels.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
    Returns:
        accuracy: [Float] of length 1
    """
    pred  = np.argmax(logits, axis = 1)
    truth = np.argmax(labels, axis = 1)
    match = np.equal(pred, truth)
    return [np.mean(match)]

def ECE(logits,
        labels,
        bins_calibration = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **params):
    """Calculate ECE from logits and labels.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
        bins_calibration: list of floats, limits of calibration bins
    Returns:
        accuracy: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis = 1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    # get the highest predicted probability in each sample
    max_prob = np.ndarray.max(probs, axis = 1)
    # calculate ECE and accumulate
    ece = calib.calculate_expected_calibration_error(
        logits, labels, bins_calibration
    )
    return [ece]

def neg_log_likelihood(logits, labels, **params):
    """Calculate negative log likelihood loss from logits and labels.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
        bins_calibration: list of floats, limits of calibration bins
    Returns:
        negative log likelihood: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis = 1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    return [log_loss(labels, probs)]

def mean_entropy(logits, labels = None, **params):
    """Calculate entropy from logits.
    The argument `labels` is unused, but kept for consistency with other
    measures.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
    Returns:
        entropy: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis = 1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    entrops = [-np.sum(np.multiply(prob, np.log(prob))) for prob in probs]
    return [np.mean(entrops)]

def brier_score(logits, labels, **params):
    """Calculate brier score from logits and labels.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
    Returns:
        brier score: [Float] of length 1
    """
    # transform logits to probabilities and clip values for numerical stability
    probs = softmax(logits, axis = 1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    # transpose and form label-prob pairs in each dimension
    lp_pairs = zip(np.transpose(labels), np.transpose(probs))
    # calculate brier score in each dimension
    brier_scores = [brier_score_loss(label, prob) for label, prob in lp_pairs]
    # sum over all dimensions
    return [np.sum(brier_scores)]

def confidence_scores(logits, labels = None, **params):
    """Calculate confidence scores from logits.
    The argument `labels` is unused, but kept for consistency with other
    measures.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
    Returns:
        confidence scores: [Float]
    """
    probs = softmax(logits, axis = 1)
    probs = np.clip(probs, 1.17e-37, 3.40e37)
    return np.max(probs, axis = 1)

def matches(logits, labels, **params):
    """Calculate matches from logits and labels.
    Args:
        logits: array, (samples, logits)
        labels: array, (samples, one hot labels)
    Returns:
        matches: [Float]
    """
    pred = np.argmax(logits, axis = 1)
    truth = np.argmax(labels, axis = 1)
    return np.equal(pred, truth)
