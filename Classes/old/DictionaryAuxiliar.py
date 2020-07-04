# Linear algebra library.
import numpy as np


def key_with_maxval(d):
    """Get the key with maximum value in a dictionary
    Paramerers:
        d (dictionary)
    """
    try:
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]
    except:
        return 9999


def key_with_minval(d):
    """Get the key with minimum value in a dictionary
    Paramerers:
        d (dictionary)
    """
    try:
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(min(v))]
    except:
        return 9999


