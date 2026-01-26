import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """
    # Create a boolean array where predictions match ground truth
    correct_predictions = (LPred == LTrue)
    
    # Accuracy is the average of matches (number of True / total)
    acc = np.mean(correct_predictions)
    
    return acc



def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """
    # Find the unique labels to determine the size of the matrix (e.g., 10 for OCR)
    labels = np.unique(LTrue)
    nClasses = len(labels)
    
    # Initialize an empty matrix of zeros
    cM = np.zeros((nClasses, nClasses))
    
    # Fill the matrix: for every sample, increment the cell (pred, true)
    for i in range(len(LTrue)):
        pred = LPred[i]
        true = LTrue[i]
        cM[pred, true] += 1
        
    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """
    # The sum of the diagonal is the total number of correct predictions
    correct = np.trace(cM)
    
    # The sum of all elements in the matrix is the total number of samples
    total = np.sum(cM)
    
    # Prevent division by zero just in case
    acc = correct / total if total > 0 else 0.0
    
    return acc
