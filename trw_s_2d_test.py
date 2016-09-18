#!env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import skimage.io
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

def compute_energy(unary, pairwise, label):
    # compute the labeling energy.

    # unary energy
    energy = unary[:, :, label].sum()

    # pairwise energy
    H, W, N_DISP = unary.shape
    pairwise_count = np.zeros((N_DISP, N_DISP), np.int32)
    for y in range(H):
        for x in range(W):
            l = label[y, x]
            if y > 0: pairwise_count[l, int(label[y - 1, x])] += 1
            if x > 0: pairwise_count[l, int(label[y, x - 1])] += 1
    energy += (pairwise * pairwise_count).sum()

    return energy


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
    # TRW-S min-sum
    H, W, N_DISP = unary.shape
    L, R, U, D = 0, 1, 2, 3
    msg = np.zeros((H, W, N_DISP, 4), np.float32)
    Ebound = 0
    gamma = 1.0 / 4 # 4 edge connected to 1 node

    # monotonic chain with an ordering and its reversed version.
    original_ordering = [(y, x) for y in range(H) for x in range(W)]
    reversed_ordering = reversed(original_ordering)

    for i in range(2):
        if i % 2 == 0:
            ordering = original_ordering
        else:
            ordering = reversed_ordering

        for y, x in ordering:
            # M_(u->s)
            s = unary[y, x, :] + msg[y, x, :, U] + msg[y, x, :, L] + msg[y, x, :, D] + msg[y, x, :, R]
            delta = np.min(s)
            s -= delta
            Ebound += delta

            # M_(s->t) update and normalize
            if i % 2 == 0 and y + 1 < H: # to down
                v = np.min((gamma*s - msg[y, x, :, D]) + pairwise, axis=0)
                delta = np.min(v)
                msg[y + 1, x, :, U] = v - delta
                Ebound += delta
            if i % 2 == 0 and x + 1 < W: # to right
                v = np.min((gamma*s - msg[y, x, :, R]) + pairwise, axis=0)
                delta = np.min(v)
                msg[y, x + 1, :, L] = v - delta
                Ebound += delta
            if i % 2 == 1 and y > 0: # to up
                v = np.min((gamma*s - msg[y, x, :, U]) + pairwise, axis=0)
                delta = np.min(v)
                msg[y - 1, x, :, D] = v - delta
                Ebound += delta
            if i % 2 == 1 and x > 0: # to left
                v = np.min((gamma*s - msg[y, x, :, L]) + pairwise, axis=0)
                delta = np.min(v)
                msg[y, x - 1, :, R] = v - delta
                Ebound += delta

    print('Ebound = ', Ebound)
    cost = unary + msg.sum(axis=3)
    return cost.argmin(axis=2)



def trw_s_2d_stereo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shrink', type=int, default=1)
    parser.add_argument('--pairwise', default='Potts')
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--method', default='LBP')
    parser.add_argument('--show-unary', action='store_true')
    args = parser.parse_args()

    # data load
    imL = skimage.io.imread('data/imL.png')
    imR = skimage.io.imread('data/imR.png')
    BASE_DISP = 9
    N_DISP = 9
    STEP_SIZE = 2
    WINDOW_SIZE = 1
    if args.shrink > 1:
        imL = scipy.ndimage.zoom(imL, 1.0/args.shrink)
        imR = scipy.ndimage.zoom(imR, 1.0/args.shrink)
        BASE_DISP = int(BASE_DISP/args.shrink)
        N_DISP = int(N_DISP/args.shrink)
        STEP_SIZE = int(3/args.shrink)

    print(imL.shape, imL.dtype, imL.min(), imL.max())
    H, W = imL.shape[:2]

    def compute_block_diff(L, R):
        if WINDOW_SIZE == 1:
            return np.abs(L - R).mean(axis=2)
        offy = WINDOW_SIZE/2
        offx = WINDOW_SIZE/2
        u = np.zeros(L.shape[:2], np.float32)
        for y in range(offy, H - offy):
            for x in range(offx, W - offx):
                u[y, x] = ((L[y - offy : y - offy + WINDOW_SIZE, x - offx : x - offx + WINDOW_SIZE] \
                        - R[y - offy : y - offy + WINDOW_SIZE, x - offx : x - offx + WINDOW_SIZE]) ** 2).mean()
        return u

    CACHE = 'cache.npz'
    if os.path.isfile(CACHE):
        f = np.load(CACHE)
        unary = f['unary']
    else:
        # data cost
        unary = np.zeros((H, W, N_DISP), np.float32)
        for idisp in range(N_DISP):
            disp = int((idisp - BASE_DISP) * STEP_SIZE)
            print(disp)
            if disp > 0:
                unary[:, :, idisp] = compute_block_diff(imL, np.concatenate([
                    imR[:, disp:, :],
                    imR[:, [-1]*abs(disp), :]], axis=1))
            elif disp < 0:
                unary[:, :, idisp] = compute_block_diff(imL, np.concatenate([
                    imR[:, [0]*abs(disp), :],
                    imR[:, :disp, :]], axis=1))
            else:
                unary[:, :, idisp] = compute_block_diff(imL, imR)
            if args.show_unary:
                plt.imshow(unary[:, :, idisp])
                plt.show()
        np.savez(CACHE, unary=unary)

    # pairwise cost
    if args.pairwise == 'Potts':
        pairwise = 1.0 - np.eye(N_DISP, dtype=np.float32)
    if args.pairwise == 'linear':
        pairwise = np.eye(N_DISP, dtype=np.float32)
        for y in range(N_DISP):
            for x in range(N_DISP):
                pairwise[y, x] = min(4, np.abs(y - x))
    pairwise *= args.coef

    # find MAP of the MRF
    if args.method == 'local':
        print('method: Local best')
        disp = solve_local(unary, pairwise)

    if args.method == 'LBP':
        print('method: Loopy Belief Propagation')
        disp = solve_lbp(unary, pairwise)

    if args.method == 'TRW-S':
        print('method: TRW-S')
        disp = solve_trw_s(unary, pairwise)

    print('energy = {}'.format(compute_energy(unary, pairwise, disp)))

    fig, ax = plt.subplots(1, 1)
    ax.imshow(disp)
    fig.suptitle('method: {}'.format(args.method))
    fig.savefig('img_{}.png'.format(args.method))

def main():
    trw_s_2d_stereo()
    plt.show()

if __name__ == '__main__':
    main()
