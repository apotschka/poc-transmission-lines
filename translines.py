'''
Discretized NLP for transmission lines example with outer convexification.

[1] Göttlich, Potschka, Teuber. A partial outer convexification approach to
    control transmission lines, 2018,
    http://www.optimization-online.org/DB_HTML/2017/11/6312.html

Copyright 2018 Andreas Potschka, Claus Teuber

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
'''

from math import sqrt, ceil
import casadi as cas
import numpy as np
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt

def network_extended_tree():
    'Extended tree network data'
    # vertices
    V = tuple(range(11))
    # edges
    A = ((0,2),(1,4),(2,5),(2,3),(2,4),(4,5),(3,6),(3,7),(4,8),(4,9),(5,10))
    # producer nodes (given as indices of V) and upper bounds
    producers = (0, 1)
    bounds = (120., 80.)
    # consumer nodes (given as indices of V)
    consumers = (6, 7, 10, 8, 9)
    # configurations (given as indices of switched off edges)
    configs = ((), (4,), (5,), (4,5))
    return V, A, producers, bounds, consumers, configs

def network_subgrid():
    'Subgrid network data'
    # vertices
    V = tuple(range(14))
    # edges
    A = ((0,2),(1,3),(3,5),(2,4),(2,8),(4,6),(5,6),(6,7),(8,7),(2,9),(8,10),
            (3,11),(7,12),(7,13))
    # producer nodes (given as indices of V) and upper bounds
    producers = (0, 1)
    bounds = (120., 80.)
    # consumer nodes (given as indices of V)
    consumers = (9, 10, 11, 12, 13)
    # configurations (given as indices of switched off edges)
    configs = ((), (2,3,7), (8,), (2,3,7,8))
    return V, A, producers, bounds, consumers, configs

def in_edges(A):
    'Compute lists of incoming edges per edge'
    delta = [[] for ij in A]
    for line, (i,j) in enumerate(A):
        delta[line] += [k for k, (l,m) in enumerate(A) if m == i]
    return delta

def out_edges(A):
    'Compute lists of outgoing edges per edge'
    delta = [[] for ij in A]
    for line, (i,j) in enumerate(A):
        delta[line] += [k for k, (l,m) in enumerate(A) if l == j]
    return delta

def equal_distribution_matrix(delta, off):
    '''Create equal distribution matrix from list of incoming/outgoing edges
    delta and configuration off'''
    n_edges = len(delta)
    # for incoming edges
    D = dok_matrix((n_edges, n_edges))
    for i in range(n_edges):
        incident = [j for j in delta[i] if not j in off]
        n_incident = len(incident)
        entry = 1. / n_incident if n_incident > 0 else 0
        for j in incident:
            D[j, i] = entry
    return D

def translines_nlp(net, demand, T=15., lr=1., L=1., C=1., R=1e-3, G=2e-3,
        nx=10, nt=151):
    'Compose NLP from network and other data'
    dx = lr / nx
    dt = T / (nt - 1)

    assert dt <= sqrt(L*C) * dx, 'Violation of CFL condition'

    b11 = 0.5 * (R/L + G/C)
    b21 = 0.5 * (R/L - G/C)
    b12 = b21
    b22 = b11

    lambda_p = 1./sqrt(L*C)
    lambda_m = -lambda_p

    # get network data
    V, A, producers, ctrl_bounds, consumers, configs = net()
    n_nodes = len(V)
    n_edges = len(A)
    n_ctrls = len(producers)
    n_confg = len(configs)

    delta_in = in_edges(A)
    delta_out = out_edges(A)
    D_p, D_m = [], []
    for off in configs:
        D_p += [equal_distribution_matrix(delta_out, off)]
        D_m += [equal_distribution_matrix(delta_in, off)]

    # start with empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    v_indices = []

    # convex configuration multipliers (partial outer convexification)
    alpha = cas.MX.sym('alpha', nt-1, n_confg)
    w += [cas.reshape(alpha, -1, 1)]
    w0 += [.5] * (nt-1) * n_confg
    lbw += [0.] * (nt-1) * n_confg
    ubw += [2.] * (nt-1) * n_confg

    # SOS1 constraint for alpha
    g += [cas.mtimes(alpha, cas.DM.ones(n_confg))]
    lbg += [1.] * (nt-1)
    ubg += [1.] * (nt-1)

    # inflow control
    u = cas.MX.sym('u', nt-1, n_ctrls)
    w += [cas.reshape(u, -1, 1)]
    for i in range(n_ctrls):
        w0 += [0.] * (nt-1)
        lbw += [0.] * (nt-1)
        ubw += [ctrl_bounds[i]] * (nt-1)

    # characteristic variables
    xi_p, xi_m = [], []
    for line in range(n_edges):
        xi_p += [cas.MX.sym('xi_p_{}_{}'.format(line, A[line]), nx, nt)]
        xi_m += [cas.MX.sym('xi_m_{}_{}'.format(line, A[line]), nx, nt)]
        w += [cas.reshape(xi_p[line], -1, 1), cas.reshape(xi_m[line], -1, 1)]
        w0 += [0.] * (2 * nx * nt)
        lbw += ([0.] * nx + [-cas.inf] * (nx * (nt-1))) * 2
        ubw += ([0.] * nx + [+cas.inf] * (nx * (nt-1))) * 2

    # dynamics on network edges
    for line in range(n_edges):
        # start and end vertex
        start_vertex, end_vertex = A[line]

        # dynamic equations (upwind discretization)
        for t in range(nt-1):
            # xi_p variables
            if start_vertex in producers:
                theta = u[t, start_vertex]
            else:
                theta = sum(alpha[t,conf] * D_p[conf][line,j] * xi_p[j][-1,t]
                        for j in delta_in[line] for conf in range(n_confg))
            theta = cas.vertcat(theta, xi_p[line][:-1,t])
            adv = -lambda_p * (xi_p[line][:,t] - theta) / dx
            src = -b11 * xi_p[line][:,t] - b12 * xi_m[line][:,t]
            g += [xi_p[line][:,t+1] - xi_p[line][:,t] - dt * (adv+src)]
            lbg += [0.,] * nx
            ubg += [0.,] * nx
            # xi_m variables
            if end_vertex in consumers:
                theta = 0
            else:
                theta = sum(alpha[t,conf] * D_m[conf][line,j] * xi_m[j][0,t]
                        for j in delta_out[line] for conf in range(n_confg))
            theta = cas.vertcat(xi_m[line][1:,t], theta)
            adv = lambda_m * (xi_m[line][:,t] - theta) / dx
            src = -b21 * xi_p[line][:,t] - b22 * xi_m[line][:,t]
            g += [xi_m[line][:,t+1] - xi_m[line][:,t] - dt * (adv+src)]
            lbg += [0.,] * nx
            ubg += [0.,] * nx

    # objective
    end_vertices = [j for (i,j) in A]
    for i, consumer in enumerate(consumers):
        line = end_vertices.index(consumer)
        # box rule (consistent with [1] up to a scaling with dt)
        for t in range(nt-1):
            J = J + 0.5 * dt * (xi_p[line][-1,t] - demand[i,t])**2

    # create NLP dictionary
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g)

    return nlp, lbw, ubw, lbg, ubg, w0

def extract_solution(sol, net, nt, nx):
    'Extract NumPy arrays from CasADi NLP solution'
    _, A, producers, _, _, configs = net()
    n_ctrls = len(producers)
    n_confg = len(configs)
    offset = 0
    alpha = np.array(cas.reshape(sol['x'][offset:offset+(nt-1)*n_confg],
        nt-1, n_confg))
    offset += (nt - 1) * n_confg
    u = np.array(cas.reshape(sol['x'][offset:offset+(nt-1)*n_ctrls],
        nt-1, n_ctrls))
    offset += (nt - 1) * n_ctrls
    xi_p, xi_m = [], []
    for line in range(len(A)):
        xi_p += [np.array(cas.reshape(sol['x'][offset:offset+nt*nx], nx, nt))]
        offset += nt*nx
        xi_m += [np.array(cas.reshape(sol['x'][offset:offset+nt*nx], nx, nt))]
        offset += nt*nx
    return u, alpha, xi_p, xi_m

def plot_solution(net, u, alpha, xi_p, xi_m, demand, T, nt):
    'Plot extracted solution'
    _, A, _, _, consumers, _ = net()
    n_ctrls = u.shape[1]
    n_confg = alpha.shape[1]
    n_consumer = demand.shape[0]
    t = np.linspace(0, T, nt)

    plt.figure(1).clear()
    fig, axes = plt.subplots(n_ctrls, 1, num=1)
    axes = axes.reshape((-1,)) if n_ctrls > 1 else [axes]
    for i in range(n_ctrls):
        axes[i].step(t, np.concatenate(([np.nan], u[:,i])))
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'control $u_{}$'.format(i))

    plt.figure(2).clear()
    fig, axes = plt.subplots(n_confg+1, 1, num=2)
    axes = axes.reshape((-1,))
    for i in range(n_confg):
        axes[i].step(t, np.concatenate(([np.nan], alpha[:,i])), linewidth=3)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'multiplier $\alpha_{}$'.format(i))
    axes[-1].step(t, np.concatenate(([np.nan], alpha.dot(np.arange(n_confg)))),
            'r', linewidth=3)
    axes[-1].set_xlabel(r'time $t$')
    axes[-1].set_ylabel(r'configuration')
    axes[-1].set_ylim(-0.02, n_confg - 0.98)

    n_edges = len(xi_p)
    n = int(ceil(sqrt(n_edges)))
    m = n-1 if n*(n-1) >= n_edges else n

    plt.figure(3).clear()
    fig, axes = plt.subplots(m, n, num=3)
    axes = axes.reshape((-1,)) if m > 1 or n > 1 else [axes]
    for i in range(n_edges):
        axes[i].pcolormesh(xi_p[i], cmap='jet')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'space $x$')
        axes[i].set_title(r'$\xi_+$ on edge {}'.format(i))

    plt.figure(4).clear()
    fig, axes = plt.subplots(m, n, num=4)
    axes = axes.reshape((-1,)) if m > 1 or n > 1 else [axes]
    for i in range(n_edges):
        axes[i].pcolormesh(xi_m[i], cmap='jet')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'space $x$')
        axes[i].set_title(r'$\xi_-$ on edge {}'.format(i))

    plt.figure(5).clear()
    fig, axes = plt.subplots(n_consumer, 1, num=5)
    axes = axes.reshape((-1,)) if n_consumer > 1 else [axes]
    end_vertices = [j for (i,j) in A]
    fmt = r'demand and delivery at vertex {}'
    for i, consumer in enumerate(consumers):
        line = end_vertices.index(consumer)
        axes[i].step(t, np.concatenate(([np.nan], demand[i,:])), 'r-')
        axes[i].plot(t, xi_p[line][-1,:], 'b-')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(fmt.format(end_vertices[line]))

def sum_up_rounding(alpha):
    'Sum-Up-Rounding with SOS1 constraint on equidistant grid'
    N, modes = alpha.shape
    dt = 1.
    p = np.zeros((N, modes))
    phat = np.zeros(modes)
    unique = True
    for i in range(N):
        phat += dt * alpha[i,:]
        j = np.argmax(phat)
        if np.sum(phat == phat[j]) > 1:
            unique = False
        p[i,j] = 1.
        phat[j] -= dt
    if not unique:
        print('\nWarning: Sum-Up Rounding result not unique\n')
    return p

if __name__ == '__main__':
    if True: # extended tree network
        net = network_extended_tree
        T = 26.
        if True: # dt = dx = 0.25
            demand = np.loadtxt('demand_extended_tree.dat')
            nt = 4*26+1 
            nx = 4
        else: # dt = dx = 0.5
            demand = np.loadtxt('demand_extended_tree_coarse.dat')
            nt = 2*26+1 
            nx = 2
    else:
        net = network_subgrid
        T = 26.
        if True: # dt = dx = 0.25
            demand = np.loadtxt('demand_subgrid.dat')
            nt = 4*26+1 
            nx = 4
        else: # dt = dx = 0.5
            demand = np.loadtxt('demand_subgrid_coarse.dat')
            nt = 2*26+1 
            nx = 2

    nlp, lbw, ubw, lbg, ubg, w0 = translines_nlp(net, demand, T, nt=nt, nx=nx)

    options = {'ipopt': {'tol': 1e-8}}
    solver = cas.nlpsol('solver', 'ipopt', nlp, options);
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    u, alpha, xi_p, xi_m = extract_solution(sol, net, nt, nx)

    if True: # Sum-up rounding and reoptimization step
        beta = sum_up_rounding(alpha)

        # reoptimize continuous controls for fixed integer variables
        for i, v in enumerate(np.reshape(beta, -1, order='F')):
            lbw[i] = v
            ubw[i] = v
        sol2 = solver(x0=sol['x'], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        u, alpha2, xi_p, xi_m = extract_solution(sol2, net, nt, nx)
    else:
        alpha2 = alpha

    plot_solution(net, u, alpha2, xi_p, xi_m, demand, T, nt)

