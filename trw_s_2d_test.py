#!env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import skimage.io
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

CACHE_PATH = 'cache.npz'

def solve_local(unary, pairwise):
    # local argmin energy. no pairwise term.
    H, W, N_DISP = unary.shape
    assert N_DISP == pairwise.shape[0] == pairwise.shape[1]

    return unary.argmin(axis=2)

def solve_lbp(unary, pairwise):
    # Loopy BP min-sum
    H, W, N_DISP = unary.shape
    L, R, U, D = 0, 1, 2, 3
    msg = np.zeros((H, W, N_DISP, 4), np.float32)

    for i in range(2): # iter
        for idir in [L, U, R, D] if i % 2 == 0 else [U, R, D, L]:
            for k in range(N_DISP):
                min_val = np.zeros((H, W), np.float32) + 1e9
                for j in range(N_DISP):
                    val = unary[:, :, j] + pairwise[k, j]
                    if idir != L: val += np.concatenate([msg[:, [0], j, L], msg[:, :-1, j, L]], axis=1)
                    if idir != R: val += np.concatenate([msg[:, 1:, j, R], msg[:, [-1], j, R]], axis=1)
                    if idir != U: val += np.concatenate([msg[[0], :, j, U], msg[:-1, :, j, U]], axis=0)
                    if idir != D: val += np.concatenate([msg[1:, :, j, D], msg[[-1], :, j, D]], axis=0)
                    min_val = np.minimum(min_val, val)
                msg[:, :, k, idir] = min_val

    cost = unary + msg.sum(axis=3)
    return cost.argmin(axis=2)

def solve_trw_s(unary, pairwise):
    # TRW-S with diagonal updating.
    H, W, N_DISP = unary.shape
    L, R, U, D = 0, 1, 2, 3
    msg = np.zeros((H, W, N_DISP, 4), np.float32)
    Ebound = 0
    gamma = 0.25

    # monotonic chain with an ordering and its reversed version.
    original_ordering = []
    for y in range(H):
        for x in range(W):
            original_ordering.append((y, x))
    reversed_ordering = reversed(original_ordering)
    for i in range(4):
        if i % 2 == 0:
            ordering = original_ordering
        else:
            ordering = reversed_ordering

        for y, x in ordering:
            s = unary[y, x, :]
            # sum_(u,s){M_(u->s)}
            if i % 2 == 0:
                if y > 0: s += msg[y - 1, x, :, U]
                if x > 0: s += msg[y, x - 1, :, L]
            else:
                if y + 1 < H: s += msg[y + 1, x, :, D]
                if x + 1 < W: s += msg[y, x + 1, :, R]
            delta = np.min(s)
            s -= delta
            Ebound += delta

            # M_(s->t) update and normalize
            if i % 2 == 0 and y + 1 < H: # to down
                v = np.zeros(N_DISP, np.float32) + 1e9
                for j in range(N_DISP):
                    v = np.minimum(v, gamma*s[j] - msg[y, x, j, D] + pairwise[j, :])
                delta = np.min(v)
                msg[y + 1, x, :, U] = v - delta
                Ebound += delta
            if i % 2 == 0 and x + 1 < W: # to right
                v = np.zeros(N_DISP, np.float32) + 1e9
                for j in range(N_DISP):
                    v = np.minimum(v, gamma*s[j] - msg[y, x, j, R] + pairwise[j, :])
                delta = np.min(v)
                msg[y, x + 1, :, L] = v - delta
                Ebound += delta
            if i % 2 == 1 and y > 0: # to up
                v = np.zeros(N_DISP, np.float32) + 1e9
                for j in range(N_DISP):
                    v = np.minimum(v, gamma*s[j] - msg[y, x, j, U] + pairwise[j, :])
                delta = np.min(v)
                msg[y - 1, x, :, D] = v - delta
                Ebound += delta
            if i % 2 == 1 and x > 0: # to left
                v = np.zeros(N_DISP, np.float32) + 1e9
                for j in range(N_DISP):
                    v = np.minimum(v, gamma*s[j] - msg[y, x, j, L] + pairwise[j, :])
                delta = np.min(v)
                msg[y, x - 1, :, R] = v - delta
                Ebound += delta

    print('Ebound = ', Ebound)
    cost = unary + msg.sum(axis=3)
    return cost.argmin(axis=2)



def trw_s_2d_stereo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shrink', type=int, default=1)
    parser.add_argument('--method', default='LBP')
    args = parser.parse_args()

    imL = skimage.io.imread('data/imL.png')
    imR = skimage.io.imread('data/imR.png')
    BASE_DISP = 9
    N_DISP = 9
    STEP_SIZE = 3
    if args.shrink > 1:
        imL = scipy.ndimage.zoom(imL, 1.0/args.shrink)
        imR = scipy.ndimage.zoom(imR, 1.0/args.shrink)
        BASE_DISP = int(BASE_DISP/args.shrink)
        N_DISP = int(N_DISP/args.shrink)
        STEP_SIZE = int(3/args.shrink)

    print(imL.shape, imL.dtype, imL.min(), imL.max())
    H, W = imL.shape[:2]

    if False and os.path.isfile(CACHE_PATH):
        f = np.load(CACHE_PATH)
        unary = f['unary']
        pairwise = f['pairwise']
        print('loaded from cache')
    else:
        # data cost
        unary = np.zeros((H, W, N_DISP), np.float32)
        for idisp in range(N_DISP):
            disp = int((idisp - BASE_DISP) * STEP_SIZE)
            print(disp)
            if disp > 0:
                unary[:, :, idisp] = np.abs(imL - np.concatenate([
                    imR[:, disp:, :],
                    imR[:, [-1]*abs(disp), :]], axis=1)).mean(axis=2)
            elif disp < 0:
                unary[:, :, idisp] = np.abs(imL - np.concatenate([
                    imR[:, [0]*abs(disp), :],
                    imR[:, :disp, :]], axis=1)).mean(axis=2)
            else:
                unary[:, :, idisp] = np.abs(imL - imR).mean(axis=2)
            #plt.imshow(unary[:, :, idisp])
            #plt.show()
        # pairwise cost = potts
        pairwise = 1.0 - np.eye(N_DISP, dtype=np.float32)
        # pairwise cost = truncated linear
        for y in range(N_DISP):
            for x in range(N_DISP):
                pairwise[y, x] = min(4, np.abs(y - x))
        np.savez(CACHE_PATH, unary=unary, pairwise=pairwise)
        print('saved to cache')

    if args.method == 'local':
        print('method: Local best')
        disp = solve_local(unary, pairwise)

    if args.method == 'LBP':
        print('method: Loopy Belief Propagation')
        disp = solve_lbp(unary, pairwise)

    if args.method == 'TRW-S':
        print('method: TRW-S')
        disp = solve_trw_s(unary, pairwise)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(disp)

def main():
    trw_s_2d_stereo()
    plt.show()

if __name__ == '__main__':
    main()
