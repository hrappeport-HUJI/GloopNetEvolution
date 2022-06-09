import time

import numpy as np
from numba import jit, prange, cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from util import time_func
from math import ceil
from scalene import scalene_profiler

G_IN_ACT = 0
G_IN_INH = 1
G_OUT_ACT = 2

#TODO can device functions be inlined?

@cuda.jit(device=True)
def get_r_g(n, t, topology_i, realization_i, g_i, g_in_act_idx, g_in_inh_idx, norm_p, norm_m, thresholds):
    r_g = 1
    for i in g_in_act_idx:
        r_g *= max(n[topology_i, realization_i, i, t] - thresholds[topology_i, g_i], 0) / norm_p[topology_i, i]
    for i in g_in_inh_idx:
        r_g *= max(1 - n[topology_i, realization_i, i, t] / norm_m[topology_i, i], 0)
    return r_g

@cuda.jit(device=True)
def gate_update(n, t, topology_i, realization_i, g_in_act_idx, g_out_act_idx):
    for i in g_in_act_idx:
        n[topology_i, realization_i, i, t] -= 1
    for i in g_out_act_idx:
        n[topology_i, realization_i, i, t] += 1


# # Stable: Before local memory optimizations
# @cuda.jit
# def run_petri_net(n, T, gate_in_out_idx, gate_s_e_idx, source_idx, sink_idx,
#                   source_vals, sink_vals, gate_order, norm_p, norm_m, thresholds, rng_states):
#     """
#     :param n: The value of each place in all realizations per each time step.
#     n[:, :, :, 0] are the initial conditions (expected to be present)
#     shape = (n_topologies, n_realizations_per_topology, n_places, T)
#     :param T: Simulation time (# iterations)
#
#     Topology parameters:
#     :param gate_in_out_idx: For each topology, 3 lists of place idx (For incoming-activators, incoming-inhibitors
#     and outgoing-activators). These are contiguous lists and gate_s_e_idx are required to delimitate them
#     shape = (n_topologies, 3, n_max_edges)  #TODO probably not necessary to max_edges
#     :param gate_s_e_idx: For each topology, 3 lists of delimiters for using gate_in_out_idx
#     shape = (n_topologies, 3, n_gates, 2)
#     :param source_idx: indices of sources
#     shape = (n_topologies, n_sources)
#     :param sink_idx: indices of sources
#     shape = (n_topologies, n_sinks)
#     Note that gate_in_out_idx, gate_s_e_idx, source_idx and sink_idx together define all topologies
#
#     :param source_vals: The value to set the sources with at each time t
#     shape = T
#     :param sink_vals: The value to set the sinks to at each time t
#     shape = T
#     :param gate_order: A permutation of gate indices for each time step  #TODO may need to generate on the fly
#     shape = (T, n_gates)
#     :param norm_p: Parameter for normalizing the gate rates
#     shape = (n_topologies, n_places)
#     :param norm_m: Parameter for normalizing the gate rates
#     shape = (n_topologies, n_places)
#     :param thresholds: Parameter for thresholding the excitatory gate inputs
#     shape = (n_topologies, n_gates)
#
#     :param rng_states:
#
#     :return: None. (n is filled with simulation data)
#     """
#     n_topologies, n_realizations_per_topology, n_places, T = n.shape
#     topology_i = cuda.blockIdx.x
#     realization_i = cuda.threadIdx.x
#     if (topology_i >= n_topologies) or (realization_i >= n_realizations_per_topology): return  # TODO assert inside of grid
#     thread_id = cuda.grid(1)
#
#     # Setup local block (topology specific) memory
#
#     # gate_idx = cuda.shared.array(shape=gate_in_out_idx.shape, dtype=gate_in_out_idx.dtype)  # block specific
#     # gate_in_out_idx
#
#     def get_g_idx(g_i):
#         g_i_in_act_s, g_i_in_act_e = gate_s_e_idx[topology_i, G_IN_ACT, g_i]
#         g_i_in_inh_s, g_i_in_inh_e = gate_s_e_idx[topology_i, G_IN_INH, g_i]
#         g_i_out_act_s, g_i_out_act_e = gate_s_e_idx[topology_i, G_OUT_ACT, g_i]
#
#         g_in_act_idx = gate_in_out_idx[topology_i, G_IN_ACT, g_i_in_act_s:g_i_in_act_e]
#         g_in_inh_idx = gate_in_out_idx[topology_i, G_IN_INH, g_i_in_inh_s:g_i_in_inh_e]
#         g_out_act_idx = gate_in_out_idx[topology_i, G_OUT_ACT, g_i_out_act_s:g_i_out_act_e]
#
#         return g_in_act_idx, g_in_inh_idx, g_out_act_idx
#
#
#     for t in range(1, T):
#         for i in range(n_places):
#             n[topology_i, realization_i, i, t] = n[topology_i, realization_i, i, t - 1]
#         # Drain sinks and fill sources
#         for source_i in source_idx[topology_i]:
#             n[topology_i, realization_i, source_i, t] = source_vals[t]
#         for sink_i in sink_idx[topology_i]:
#             n[topology_i, realization_i, sink_i, t] = sink_vals[t]
#         # Iterate over gates
#         for g_i in gate_order[t]:
#             g_in_act_idx, g_in_inh_idx, g_out_act_idx = get_g_idx(g_i)
#             r_g = get_r_g(n, t, topology_i, realization_i, g_i, g_in_act_idx, g_in_inh_idx, norm_p, norm_m, thresholds)
#
#
#             if xoroshiro128p_uniform_float32(rng_states, thread_id) < r_g:
#                 gate_update(n, t, topology_i, realization_i, g_in_act_idx, g_out_act_idx)


@cuda.jit
def run_petri_net(n, T, gate_in_out_idx, gate_s_e_idx, source_idx, sink_idx,
                  source_vals, sink_vals, gate_order, norm_p, norm_m, thresholds, rng_states):
    """
    :param n: The value of each place in all realizations per each time step.
    n[:, :, :, 0] are the initial conditions (expected to be present)
    shape = (n_topologies, n_realizations_per_topology, n_places, T)
    :param T: Simulation time (# iterations)

    Topology parameters:
    :param gate_in_out_idx: For each topology, 3 lists of place idx (For incoming-activators, incoming-inhibitors
    and outgoing-activators). These are contiguous lists and gate_s_e_idx are required to delimitate them
    shape = (n_topologies, 3, n_max_edges)  #TODO probably not necessary to max_edges
    :param gate_s_e_idx: For each topology, 3 lists of delimiters for using gate_in_out_idx
    shape = (n_topologies, 3, n_gates, 2)
    :param source_idx: indices of sources
    shape = (n_topologies, n_sources)
    :param sink_idx: indices of sources
    shape = (n_topologies, n_sinks)
    Note that gate_in_out_idx, gate_s_e_idx, source_idx and sink_idx together define all topologies

    :param source_vals: The value to set the sources with at each time t
    shape = T
    :param sink_vals: The value to set the sinks to at each time t
    shape = T
    :param gate_order: A permutation of gate indices for each time step  #TODO may need to generate on the fly
    shape = (T, n_gates)
    :param norm_p: Parameter for normalizing the gate rates
    shape = (n_topologies, n_places)
    :param norm_m: Parameter for normalizing the gate rates
    shape = (n_topologies, n_places)
    :param thresholds: Parameter for thresholding the excitatory gate inputs
    shape = (n_topologies, n_gates)

    :param rng_states:

    :return: None. (n is filled with simulation data)
    """



    n_topologies, n_realizations_per_topology, n_places, T = n.shape
    topology_i = cuda.blockIdx.x
    realization_i = cuda.threadIdx.x
    if (topology_i >= n_topologies) or (
            realization_i >= n_realizations_per_topology): return  # TODO assert inside of grid
    thread_id = cuda.grid(1)

    # Setup local block (topology specific) memory

    def copy_local_gate_in_out_idx(gate_in_out_idx, local_gate_in_out_idx, topology_i):
        m, n = local_gate_in_out_idx.shape
        for i in range(m):
            for j in range(n):
                local_gate_in_out_idx[i, j] = gate_in_out_idx[topology_i, i, j]

    def copy_local_gate_s_e_idx(gate_s_e_idx, local_gate_s_e_idx, topology_i):
        m, n, l = local_gate_s_e_idx.shape
        for i in range(m):
            for j in range(n):
                for k in range(l):
                    local_gate_s_e_idx[i, j, k] = gate_s_e_idx[topology_i, i, j, k]

    _, _, n_max_edges = gate_in_out_idx.shape
    _, _, n_gates, _ = gate_s_e_idx.shape
    local_gate_in_out_idx = cuda.shared.array(shape=(3, 1500), dtype=gate_in_out_idx.dtype)  # block specific
    local_gate_s_e_idx = cuda.shared.array(shape=(3, 500, 2), dtype=gate_s_e_idx.dtype)  # block specific
    copy_local_gate_in_out_idx(gate_in_out_idx, local_gate_in_out_idx, topology_i)
    copy_local_gate_s_e_idx(gate_s_e_idx, local_gate_s_e_idx, topology_i)


    cuda.syncthreads()



    def get_g_idx(g_i):
        g_i_in_act_s, g_i_in_act_e = local_gate_s_e_idx[G_IN_ACT, g_i]
        g_i_in_inh_s, g_i_in_inh_e = local_gate_s_e_idx[G_IN_INH, g_i]
        g_i_out_act_s, g_i_out_act_e = local_gate_s_e_idx[G_OUT_ACT, g_i]

        g_in_act_idx = local_gate_in_out_idx[G_IN_ACT, g_i_in_act_s:g_i_in_act_e]
        g_in_inh_idx = local_gate_in_out_idx[G_IN_INH, g_i_in_inh_s:g_i_in_inh_e]
        g_out_act_idx = local_gate_in_out_idx[G_OUT_ACT, g_i_out_act_s:g_i_out_act_e]

        return g_in_act_idx, g_in_inh_idx, g_out_act_idx

    for t in range(1, T):
        for i in range(n_places):
            n[topology_i, realization_i, i, t] = n[topology_i, realization_i, i, t - 1]
        # Drain sinks and fill sources
        for source_i in source_idx[topology_i]:
            n[topology_i, realization_i, source_i, t] = source_vals[t]
        for sink_i in sink_idx[topology_i]:
            n[topology_i, realization_i, sink_i, t] = sink_vals[t]
        # Iterate over gates
        for g_i in gate_order[t]:
            g_in_act_idx, g_in_inh_idx, g_out_act_idx = get_g_idx(g_i)
            r_g = get_r_g(n, t, topology_i, realization_i, g_i, g_in_act_idx, g_in_inh_idx, norm_p, norm_m, thresholds)

            if xoroshiro128p_uniform_float32(rng_states, thread_id) < r_g:
                gate_update(n, t, topology_i, realization_i, g_in_act_idx, g_out_act_idx)

@cuda.jit(device=True)
def copy(a, b, idx):
    for i in idx:
        b[i] = a[i]

@cuda.jit
def cuda_test(a, b, idx):
    def first_half(x):
        return x[:len(x)//2]
    thread_id = cuda.grid(1)
    copy(a[:, thread_id], b[:, thread_id], first_half(idx))

@time_func
def main():

    rng_seed = 1
    np.random.seed(rng_seed)
    #  ------------------------------ Random large scale nets ------------------------
    n_topologies = 128
    n_realizations_per_topology = 128
    T = 500
    n_places = 100
    n_gates = 500
    n_edges = 1500
    n_sources = 5
    n_sinks = 5
    gate_in_out_idx = cuda.to_device(np.random.randint(0, n_places, size=(n_topologies, 3, n_edges), dtype=np.int32))
    gate_s_e_idx = cuda.to_device(np.random.randint(0, n_places, size=(n_topologies, 3, n_gates, 2), dtype=np.int32))
    source_idx = cuda.to_device(np.random.randint(0, n_places, size=(n_topologies, n_sources), dtype=np.int32))
    sink_idx = cuda.to_device(np.random.randint(0, n_places, size=(n_topologies, n_sinks), dtype=np.int32))
    #  ------------------------------ \Random large scale nets ------------------------

    #  ------------------------------ Toy topologies (2 topologies of 5 places and 2 gates) ------------------------
    # n_topologies = 2
    # n_realizations_per_topology = 2
    # n_places = 5
    # n_gates = 2
    # T = 10
    #
    # gate_in_out_idx = cuda.device_array(shape=(n_topologies, 3, 4), dtype=np.int32)
    # gate_s_e_idx = cuda.device_array(shape=(n_topologies, 3, n_gates, 2), dtype=np.int32)
    # source_idx = cuda.device_array(shape=(n_topologies, 1), dtype=np.int32)
    # sink_idx = cuda.device_array(shape=(n_topologies, 1), dtype=np.int32)
    # #
    # gate_in_out_idx[0, G_IN_ACT, :4] = [0, 1,   2, 3]
    # # gate_in_out_idx[0, G_IN_INH] = []
    # gate_in_out_idx[0, G_OUT_ACT, :3] = [2, 3,   4]
    #
    # gate_in_out_idx[1, G_IN_ACT, :4] = [0,   1, 2, 3]
    # # gate_in_out_idx[1, G_IN_INH] = []
    # gate_in_out_idx[1, G_OUT_ACT, :2] = [2,   4]
    #
    # gate_s_e_idx[0, G_IN_ACT] = [[0, 2], [2, 4]]
    # # gate_s_e_idx[0, G_IN_INH] = []
    # gate_s_e_idx[0, G_OUT_ACT] = [[0, 2], [2, 3]]
    #
    # gate_s_e_idx[1, G_IN_ACT] = [[0, 1], [1, 4]]
    # # gate_s_e_idx[1, G_IN_INH] = []
    # gate_s_e_idx[1, G_OUT_ACT] = [[0, 1], [1, 2]]
    #
    # source_idx[:, 0] = [0, 0]
    # sink_idx[:, 0] = [4, 4]
    #  ------------------------------ \Toy topologies ------------------------

    BPG, TPB = n_topologies, n_realizations_per_topology
    rng_states = create_xoroshiro128p_states(n_topologies*n_realizations_per_topology, seed=rng_seed)


    n = cuda.device_array(shape=(n_topologies, n_realizations_per_topology, n_places, T), dtype=np.int32)
    n[:, :, :, 0] = np.random.randint(0, 4, size=(n_topologies, n_realizations_per_topology, n_places), dtype=np.int32)


    source_vals = cuda.to_device(np.full(T, fill_value=5))
    sink_vals = cuda.to_device(np.full(T, fill_value=0))

    gate_order = cuda.to_device(np.stack([np.random.permutation(n_gates) for _ in range(T)]))
    print(gate_order)
    norm_p = cuda.to_device(np.full(shape=(n_topologies, n_places), fill_value=100, dtype=np.int32))
    norm_m = cuda.to_device(np.full(shape=(n_topologies, n_places), fill_value=500, dtype=np.int32))
    thresholds = cuda.to_device(np.full(shape=(n_topologies, n_gates), fill_value=0, dtype=np.int32))

    # cuda.profile_start()
    run_petri_net[BPG, TPB](n, T, gate_in_out_idx, gate_s_e_idx, source_idx, sink_idx,
                      source_vals, sink_vals, gate_order, norm_p, norm_m, thresholds, rng_states)
    # cuda.profile_stop()

    n = n.copy_to_host()
    return n



if __name__ == '__main__':
    n = main()
    # np.save(f"Petri_net_results\\test_case_1.npy", n)
    # comp_n = np.load(f"Petri_net_results\\test_case_1.npy")
    # assert np.equal(n, comp_n).all()



