import numpy as np

def find_overlap(box, data):
    """Find overlap between a box and a data set"""
    overalps=[]
    indexes=[]

    if len(data) == 0:
        return np.array([]), np.array([])

    x,y,w,h = box
    x1min = x
    x1max = x+w
    y1min = y
    y1max = y+h

    for i, bb in enumerate(data):
        _x,_y,_w,_h = bb
        x2min = _x
        x2max = _x+_w
        y2min = _y
        y2max = _y+_h
        if (x1min<x2max and x2min<x1max and y1min < y2max and y2min < y1max) :
            overalps.append(bb)
            indexes.append(i)

    return np.array(overalps), np.array(indexes)