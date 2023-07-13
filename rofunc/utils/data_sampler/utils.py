#!/usr/bin/env python3

def rreplace(s, old, new, n):
    """Replace the last n occurrences of a string.
    args:
        s: string to modify
        old: string to replace
        new: string to replace with
        n: number of occurrences to replace

    returns:
        modified string
    """
    li = s.rsplit(old, n)
    return new.join(li)

def pcd_concat(pcds):
    pcd = pcds[0]
    for i in range(1, len(pcds)):
        pcd += pcds[i]
    return pcd
