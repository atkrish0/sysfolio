from operator import itemgetter
import numpy as np

def softmax_stable(scores, temp=1.0):
    scores = np.array(scores)
    scores -= np.max(scores)  # Stability trick
    exp_scores = np.exp(scores / temp)
    return exp_scores / np.sum(exp_scores)