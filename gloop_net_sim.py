import numpy as np
import util
from numba import jit, prange, cuda
from exp_log import GloopNetExpLog
import torch
from line_profiler_pycharm import profile
from itertools import zip_longest
from math import ceil
import time
import shutil
from collections import namedtuple

NP_DTYPE = np.float32
TORCH_DTYPE = torch.float32
TPB = 256  # Threads per block (for Cuda kernels) TODO may be optimizable
BLOCK_MID = TPB//2
TPB_LOG_2 = int(np.log2(TPB))


def gen_topology(n_spins, alpha=1.5, L_min=1, L_max=2_500):
    """
    Generate a topology (a list of loop sizes) according to a power law distribution on loop sizes
    :param n_spins: Number of total spins (sum(topology)==n_spins)
    :param alpha: power law exponent
    :param L_min: Minimal loop size
    :param L_max: Maximal loop size
    :return: A list of loop lengths
    """
    cur_n_spins = 0
    topology = []
    while True:
        u = np.random.uniform()
        loop_size = int(((L_max ** (1 - alpha) - L_min ** (1 - alpha)) * u + L_min ** (alpha - 1)) ** (1 / (1 - alpha)))
        if cur_n_spins + loop_size >= n_spins:
            topology.append(n_spins - cur_n_spins)
            break
        topology.append(loop_size)
        cur_n_spins += loop_size
    return topology


def get_feeding_idx(topology, contiguous_C=True, return_only_F=True):
    """
    Compute which indices feed into which others in the given topology
    :param topology: A list of loop lengths
    :param contiguous_C: Indexing scheme. Indicates whether the communicating spins are bunched together at the
           beginning or else indexing is done by loop order. Namely:
           If False, then 0 is treated as the communicating index of the first loop, followed by the other
           topology[0]-1 spins in the first loop. Index topology[0] is the communicating index of the second loop etc.
           If True, than the first len(topology) indices are considered as the communicating indices, followed by the
           non-communicating indices of the first loop, then the second, etc.
    :return: A ndarray F_idx s.t. F_idx[i] has an arrow pointing to i in the given topology according to the specified indexing scheme
    """
    # Compute indices according to a non-contiguous scheme
    n_spins = np.sum(topology)
    C_idx = np.zeros_like(topology)  # C spins (communicating)
    C_idx[1:] = np.cumsum(topology[:-1])
    mask = np.zeros(n_spins, dtype=np.bool_)
    mask[C_idx] = True
    R_idx = np.arange(n_spins)[~mask]  # R spins (Rest - Non communicating)

    FC_idx = np.array([cs_ls - 1 for cs_ls in np.cumsum(topology)])  # spins feeding into C spins
    FR_idx = R_idx - 1  # spins feeding into R spins

    if contiguous_C:  # Permute so that the C indices are at the beginning
        perm_idx = np.zeros(n_spins, dtype=np.int32)
        perm_idx[C_idx] = np.arange(len(C_idx))
        perm_idx[R_idx] = np.arange(len(C_idx), len(C_idx) + len(R_idx))
        FC_idx = perm_idx[FC_idx]
        FR_idx = perm_idx[FR_idx]

    if return_only_F:
        F_idx = np.concatenate([FC_idx, FR_idx])
        return F_idx
    else:
        return C_idx, R_idx, FC_idx, FR_idx


def get_diff_top_feeding_idx(topologies, contiguous_C=True):
    """
    Generate a F_idx (see in get_feeding_idx description) for each topology in topologies
    :param topologies: a list of n_realizations topologies *with the same number of spins*
    :param contiguous_C: see in get_feeding_idx description
    :return: A ndarray F_idx (shape=(n_realizations, n_spins))
    """
    n_realizations, n_spins = len(topologies), np.sum(topologies[0])
    F_idx = np.zeros(shape=(n_realizations, n_spins))
    for realization_i in prange(n_realizations):
        F_idx[realization_i] = get_feeding_idx(topologies[realization_i], contiguous_C=contiguous_C)
    return F_idx


@cuda.jit
def sum_spins_per_loop_cuda(loop_mag_sum, spins, topology):
    """
    Cuda kernel.
    Sums the tensor spins along the second axis according to loop lengths indicated by topology
    :param loop_mag_sum: Tensor to place result in (shape=(n_realizations, n_spins))
           i.e. once completed, loop_mag_sum[realization_i, loop_i] == sum(spins[realization_i, loop_i_idxs])
    :param spins: shape=(n_realizations, n_spins) of values in {-1, 1}
    :param topology: List of loop lengths (sum(topology)==n_spins)
    :return None (operates in-place on loop_mag_sum)
    """
    n_loops, n_realizations, n_spins = len(topology), spins.shape[0], spins.shape[1]
    pos = cuda.grid(1)
    if pos > n_spins: return
    # This is intended to figure out, based on the spin index, what loop this thread is responsible for
    if pos < len(topology):
        loop_idx = pos
    else:
        loop_idx = 0
        cur_spin_idx = n_loops
        while True:
            cur_loop_size = topology[loop_idx]
            if cur_spin_idx <= pos < cur_spin_idx + cur_loop_size - 1:
                break
            loop_idx += 1
            cur_spin_idx += cur_loop_size - 1

    for realization_i in range(n_realizations):
        # Atomic prevents race conditions
        cuda.atomic.add(loop_mag_sum, (realization_i, loop_idx), spins[realization_i, pos])

@cuda.jit
def sum_spins_per_loop_cuda_non_contiguous_C(loop_mag_sum, spins, spin_loop_dict):
    """
    as sum_spins_per_loop_cuda but with a non-contiguous C indexing scheme
    spin_loop_dict: Tensor of length n_spins which maps each spin to its loop idx
    """
    n_realizations, n_spins = spins.shape[0], spins.shape[1]
    pos = cuda.grid(1)
    if pos > n_spins: return
    loop_idx = spin_loop_dict[pos]
    for realization_i in range(n_realizations):
        # Atomic prevents race conditions
        cuda.atomic.add(loop_mag_sum, (realization_i, loop_idx), spins[realization_i, pos])

class MagCompute:
    """
    A logger for network magnetization in a given simulation run
    """
    def __init__(self, spins, topology):
        """
        Setup tensors and cuda handles
        :param spins: shape=(n_realizations, n_spins) of values in {-1, 1}
        :param topology: List of loop lengths (sum(topology)==n_spins)
        """
        self.spins = spins
        self.topology = topology
        self.n_loops, self.n_realizations, self.n_spins = len(topology), spins.shape[0], spins.shape[1]
        self.loop_mag_sum = torch.zeros(size=(self.n_realizations, self.n_loops), device=torch.device("cuda:0"))
        self.BPG = ceil(self.n_spins / TPB)
        cuda_spin_loop_dict = MagCompute.get_cuda_spin_loop_dict(topology)
        # Cuda handles (Creating these takes a lot of time and is therefore done once per simulation)
        self.cuda_loop_mag_sum = cuda.as_cuda_array(self.loop_mag_sum)
        self.cuda_spins = cuda.as_cuda_array(spins)
        self.cuda_spin_loop_dict = cuda_spin_loop_dict
        # self.cuda_topology = cuda.as_cuda_array(topology)

    @staticmethod
    def get_cuda_spin_loop_dict(topology):
        np_topology = topology.cpu().numpy()
        spin_loop_dict = np.zeros(shape=np.sum(np_topology), dtype=np.long)
        loop_i_s, loop_i_e = 0, 0
        for loop_i in range(len(np_topology)):
            loop_i_e += np_topology[loop_i]
            spin_loop_dict[loop_i_s: loop_i_e] = loop_i
            loop_i_s = loop_i_e
        return cuda.as_cuda_array(torch.tensor(spin_loop_dict, device=torch.device("cuda:0")))


    @profile
    def __call__(self):
        """
        Log the average magnetization (defined as the average over the averages for each loop)
        :param t: Current time step
        :return None (operates in-place on avg_mag)
        """
        # TODO CUDA all
        self.loop_mag_sum.fill_(0)
        # Blocks per grid
        # sum_spins_per_loop_cuda[self.BPG, TPB](self.cuda_loop_mag_sum, self.cuda_spins, self.cuda_topology)
        sum_spins_per_loop_cuda_non_contiguous_C[self.BPG, TPB](self.cuda_loop_mag_sum, self.cuda_spins, self.cuda_spin_loop_dict)
        self.loop_mag_sum /= self.topology  # Divide by loop lengths to get mean
        return self.loop_mag_sum
        # return torch.mean(self.loop_mag_sum[:len(self.cuda_spins)], dim=1)  # :len(self.cuda_spins) since cuda_spins may be shortened between calls


@jit(nopython=True, parallel=True)
def log_avg_mag_torch_diff_topologies(avg_mag, t, spins, topologies, n_loops_per_top):
    # TODO CUDA
    def get_top_mag(top_i, top, top_n_loops):
        top_mag = 0
        cur_R_idx = top_n_loops
        for loop_i in prange(top_n_loops):
            if top[loop_i] == 0: break
            next_R_idx = cur_R_idx + top[loop_i] - 1
            top_mag += (spins[top_i, loop_i] + np.sum(spins[top_i, cur_R_idx: next_R_idx])) / top[loop_i]
            cur_R_idx = next_R_idx
        return top_mag / top_n_loops

    for top_i in prange(len(topologies)):
        avg_mag[top_i, t] = get_top_mag(top_i, topologies[top_i], n_loops_per_top[top_i])


def create_J_C(J_C, same_J_C, J_C_sigma, n_realizations, topology, rng_seed=None):
    """
    Create a communicating-spins connection matrix.
    :param J_C: If None, a new random tensor is returned. If already has a specific value, converted to the right format and returned
    :param same_J_C: Randomize a single J_C matrix or a list of different matrices, one for each realization
    :param J_C_sigma: J_C entries (except diagonals, which are 0) are sampled from N(0, J_C_sigma)
    :param n_realizations: Number of network realizations to be run with the created J_C (relevant iff same_J_C==False)
    :param topology: List of loop lengths (sum(topology)==n_spins)
    :return: A ndarray J_C with J_C.shape=(n_loops, n_loops) if same_J_C else J_C.shape=(n_realizations, n_loops, n_loops)
    """
    n_loops = len(topology)
    if J_C is not None:  # Specific value passed. Not randomized
        J_C = J_C.astype(NP_DTYPE)
        return J_C
    elif same_J_C:  # Randomize one J_C across realizations
        with util.np_temp_seed(temp_seed=rng_seed):
            J_C = J_C_sigma * np.random.randn(n_loops, n_loops).astype(NP_DTYPE)  # Randomize J_C
        J_C[np.diag_indices(n_loops)] = 0  # Zero diagonal entries
        return J_C
    else:
        with util.np_temp_seed(temp_seed=rng_seed):
            J_C = J_C_sigma * np.random.randn(n_realizations, n_loops, n_loops)  # Randomize J_C
        for i in range(n_realizations): J_C[i][np.diag_indices(n_loops)] = 0  # Zero diagonal entries
    return J_C.astype(NP_DTYPE)


def create_J_C_diff_topologies(topologies, J_C_sigmas):
    """
    Create a list of communicating-spins connection matrices for a list of different topologies.
    :param topologies: List of topologies (see create_J_C())
    :param J_C_sigmas: Corresponding J_C_sigma to use for each topology (see create_J_C())
    :return: A ndarray J_C with J_C.shape=(n_topologies, max_n_loops, max_n_loops) zero padded for topologies with less loops
    """
    n_topologies = len(topologies)
    topology_sizes = [len(topology) for topology in topologies]
    max_n_loops = max(topology_sizes)
    J_C = np.zeros(shape=(n_topologies, max_n_loops, max_n_loops), dtype=NP_DTYPE)
    for i in range(n_topologies):
        J_C[i, :topology_sizes[i], :topology_sizes[i]] = \
            create_J_C(J_C=None, same_J_C=True, J_C_sigma=J_C_sigmas[i], n_realizations=None, topology=topologies[i])
    return J_C


def create_spins(spins, same_init_spins, spins_p, n_realizations, n_spins):
    """
    Create an array of spins (in {-1, 1}) to be used as initial conditions for a given run
    :param spins: If None, a new random tensor is returned. If already has a specific value, converted to the right format and returned
    :param same_init_spins: Randomize a single array or a list of different arrays, one for each realization
    :param spins_p: Each spin is sampled from Ber(p). If None, a different p~Uni([0, 1]) is used for each realization
    :param n_realizations: Number of network realizations to be run with the created J_C (relevant iff same_init_spins==False)
    :param n_spins: Number of spins in a single realization
    :return: A ndarray spins with spins.shape=(n_spins) if same_init_spins else spins.shape=(n_realizations, n_spins)
    """
    if spins is not None:  # Specific value passed. Not randomized
        if len(spins.shape) == 1:  # Single list of spins. Need to copy across realizations
            # Note: need to astype() first (otherwise it's copied instead of broadcasted)
            return np.broadcast_to(spins.astype(NP_DTYPE), (n_realizations, *spins.shape))
    elif same_init_spins:  # Randomize one initial conditions across realizations
        spins = np.random.choice([-1, 1], size=n_spins, p=[1 - spins_p, spins_p]).astype(NP_DTYPE)
        spins = np.broadcast_to(spins, (n_realizations, *spins.shape))  # TODO repeat instead of broadcast?
        return spins
    else:
        if spins_p is None:  # New p for each realization
            spins = np.stack([np.random.choice([-1, 1], size=n_spins, p=[1 - p_i, p_i])
                              for p_i in np.random.uniform(0, 1, n_realizations)]).astype(NP_DTYPE)
        else:  # Specific bias
            spins = np.random.choice([-1, 1], size=(n_realizations, n_spins), p=[1 - spins_p, spins_p]).astype(NP_DTYPE)
    return spins.astype(NP_DTYPE)


# @profile
@util.time_func
def _run_sim_torch(sim_len, J_C, spins, topology, H, F_idx, avg_mag=None, full_state=None):
    """
    Run one simulation with the given parameters and log results
    :param sim_len: Number of time steps
    :param J_C: tensor of connecting spins (Either with shape=(n_realizations, n_loops, n_loops) or else
                shape=(n_loops, n_loops) for using the same connection matrix for all realizations)
    :param spins: tensor of spins (in {-1, 1}) with shape=(n_realizations, n_spins)
    :param topology: List of loop lengths (sum(topology)==n_spins)
    :param H: A tensor of shape=(sim_len, n_loops) with the magnetic field each loop feels at each time point or
              otherwise of shape=(sim_len) with the magnetic field each whole network feels at each time point
    :param F_idx: A ndarray F_idx indicating which indices feed into which others in the given topology
                  i.e. F_idx[i] has an arrow pointing to i in the given topology
    :param avg_mag: A tensor with shape=(n_realizations, sim_len) to record network magnetization at each time point.
                    If None, magnetization is not recorded
    :param full_state: A tensor with shape=(n_realizations, sim_len, n_spins) to record full network state at each time point.
                    If None, full state is not recorded
    :return: None (Logging is done to the passed logging tensors avg_mag and full_state)
    """
    n_loops = len(topology)
    new_spins = torch.empty_like(spins)
    if avg_mag is not None: avg_mag_logger = MagCompute(spins, topology)
    for t in range(sim_len):
        # Logging
        if avg_mag is not None: avg_mag[:, t] = avg_mag_logger()  # TODO optimize away the two conditions
        if full_state is not None: full_state[:, t, :] = spins
        # Dynamics step
        new_spins.fill_(0)
        new_spins[:, :n_loops] += torch.matmul(J_C, spins[:, :n_loops, None]).squeeze(-1)
        new_spins += spins[:, F_idx]
        new_spins[:, :n_loops] += H[t]
        spins[...] = torch.sign(new_spins)


# # @util.time_func
def _run_sim_torch_only_lag_hist(J_C, spins, topology, H, F_idx, lag_start_time):
    """
    As _run_sim_torch, but specialized to only record lag-times (and discontinue the run of recovered nets, accelerating runtime)
    """
    sim_len, n_loops = len(H), len(topology)
    mul_J_C = len(J_C.size()) > 2
    new_spins = torch.empty_like(spins)
    mag_compute = MagCompute(spins, topology)
    lag_times_hist = torch.zeros(size=(sim_len - lag_start_time, ), dtype=torch.long, device=spins.device)

    for t in range(sim_len):
        # Remove indices of
        if t > lag_start_time:
            active_nets_idx = mag_compute() > 0
            n_recovered = len(active_nets_idx) - torch.count_nonzero(active_nets_idx)
            if n_recovered:
                lag_times_hist[t - lag_start_time] += n_recovered
                if n_recovered == len(active_nets_idx):
                    return lag_times_hist
                if mul_J_C: J_C = J_C[active_nets_idx]
                new_spins = new_spins[active_nets_idx]
                spins = spins[active_nets_idx]
                mag_compute.cuda_spins = cuda.as_cuda_array(spins)

        # Dynamics step
        new_spins.fill_(0)
        new_spins[:, :n_loops] += torch.matmul(J_C, spins[:, :n_loops, None]).squeeze(-1)
        new_spins += spins[:, F_idx]
        new_spins[:, :n_loops] += H[t]
        spins[...] = torch.sign(new_spins)

    lag_times_hist[-1] += len(spins)
    return lag_times_hist


# @profile
@util.time_func
def _run_sim_torch_only_lag_times(J_C, spins, topology, H, F_idx, lag_start_time):
    """
    As _run_sim_torch, but specialized to only record lag-times (and discontinue the run of recovered nets, accelerating runtime)
    """
    sim_len, n_loops, n_realizations, device = len(H), len(topology), len(spins), spins.device
    mul_J_C = len(J_C.size()) > 2
    new_spins = torch.empty_like(spins)
    mag_compute = MagCompute(spins, topology)
    lag_times = torch.zeros(size=(n_realizations, ), dtype=torch.int16, device=device)
    active_nets_mask = torch.full((n_realizations, ), fill_value=True, device=device)
    for t in range(sim_len):
        # Remove indices of
        if t > lag_start_time:
            crossed_idx = mag_compute() <= 0
            n_recovered = torch.count_nonzero(crossed_idx)
            if n_recovered:  # any recovered realizations
                lag_times[torch.where(active_nets_mask)[0][crossed_idx]] = t - lag_start_time
                if n_recovered == len(spins):
                    return lag_times
                active_nets_mask[torch.where(active_nets_mask)[0][crossed_idx]] = False

                if mul_J_C: J_C = J_C[~crossed_idx]
                new_spins = new_spins[~crossed_idx]
                spins = spins[~crossed_idx]
                mag_compute.cuda_spins = cuda.as_cuda_array(spins)

        # Dynamics step
        new_spins.fill_(0)
        new_spins[:, :n_loops] += torch.matmul(J_C, spins[:, :n_loops, None]).squeeze(-1)
        new_spins += spins[:, F_idx]  # TODO bottleneck
        new_spins[:, :n_loops] += H[t]
        spins[...] = torch.sign(new_spins)

    return lag_times






def FC_rewind(topology):
    cs = torch.zeros(len(topology), dtype=torch.long, device=topology.device)
    cs[1:] = torch.cumsum(topology, dim=0)[:-1]
    def FC_rewind_f(FC_idx):
        FC_idx -= cs
        FC_idx -= 1
        FC_idx %= topology
        FC_idx += cs
    return FC_rewind_f

def update_recovered_realizations(mul_J_C, mag_compute, n_realizations, lag_times, lag_start_time, device):
    active_nets_mask = torch.full((n_realizations, ), fill_value=True, device=device)
    def update_recovered_realizations_f(t, J_C, spins, loop_mag):
        loop_mag_avg = torch.mean(loop_mag[:len(spins)], dim=1)
        crossed_idx = loop_mag_avg <= 0
        n_recovered = torch.count_nonzero(crossed_idx)
        if n_recovered:  # any recovered realizations
            lag_times[torch.where(active_nets_mask)[0][crossed_idx]] = t - lag_start_time
            active_nets_mask[torch.where(active_nets_mask)[0][crossed_idx]] = False

            if mul_J_C: J_C = J_C[~crossed_idx]
            spins = spins[~crossed_idx]
            mag_compute.cuda_spins = cuda.as_cuda_array(spins)
        return J_C, spins, n_recovered == len(spins)
    return update_recovered_realizations_f


# @profile
# @util.time_func
def _run_sim_torch_only_lag_times_cyclical_shift(J_C, spins, topology, H, C_idx, FC_idx, lag_start_time):
    """
    As _run_sim_torch, but specialized to only record lag-times (and discontinue the run of recovered nets, accelerating runtime)
    :param loop_mags: A tensor with shape=(n_realizations, sim_len, n_loops) to record full network state at each time point. If None, full state is not recorded
    """
    sim_len, n_loops, n_realizations, device = len(H), len(topology), len(spins), spins.device
    lag_times = torch.zeros(size=(n_realizations, ), dtype=torch.int16, device=device)
    FC_rewind_f = FC_rewind(topology)
    mag_compute = MagCompute(spins, topology)
    update_recovered_realizations_f = update_recovered_realizations(mul_J_C=len(J_C.size()) > 2, mag_compute=mag_compute, n_realizations=n_realizations, lag_times=lag_times, lag_start_time=lag_start_time, device=device)

    for t in range(sim_len):
        # Remove indices of crossed realizations
        if t > lag_start_time:
            J_C, spins, done = update_recovered_realizations_f(t, J_C, spins, loop_mag=mag_compute())
            if done: return lag_times
        # Dynamics step
        spins[:, FC_idx] += torch.matmul(J_C, spins[:, C_idx, None]).squeeze(-1) + H[t]
        spins[:, FC_idx] = torch.sign(spins[:, FC_idx])
        C_idx = FC_idx.clone()
        FC_rewind_f(FC_idx)
    return lag_times

def _run_sim_torch_rec_cyclical_shift(J_C, spins, topology, H, C_idx, FC_idx, lag_start_time, loop_mag_rec):
    """
    As _run_sim_torch, but specialized to only record lag-times (and discontinue the run of recovered nets, accelerating runtime)
    :param loop_mags: A tensor with shape=(n_realizations, sim_len, n_loops) to record full network state at each time point. If None, full state is not recorded
    """
    sim_len, n_loops, n_realizations, device = len(H), len(topology), len(spins), spins.device
    lag_times = torch.zeros(size=(n_realizations, ), dtype=torch.int16, device=device)
    FC_rewind_f = FC_rewind(topology)
    mag_compute = MagCompute(spins, topology)
    update_recovered_realizations_f = update_recovered_realizations(mul_J_C=len(J_C.size()) > 2, mag_compute=mag_compute, n_realizations=n_realizations, lag_times=lag_times, lag_start_time=lag_start_time, device=device)

    for t in range(sim_len):
        # Remove indices of crossed realizations
        loop_mag_rec[:, t, :] = mag_compute()
        # Dynamics step
        spins[:, FC_idx] += torch.matmul(J_C, spins[:, C_idx, None]).squeeze(-1) + H[t]
        spins[:, FC_idx] = torch.sign(spins[:, FC_idx])
        C_idx = FC_idx.clone()
        FC_rewind_f(FC_idx)
    return lag_times

def run_RCCN(J_C, spins, topology, H, C_idx, FC_idx, lag_start_time, loop_mag_rec=None):
    if loop_mag_rec is None:
        return _run_sim_torch_only_lag_times_cyclical_shift(J_C, spins, topology, H, C_idx, FC_idx, lag_start_time)
    return _run_sim_torch_rec_cyclical_shift(J_C, spins, topology, H, C_idx, FC_idx, lag_start_time, loop_mag_rec)


@util.time_func
def _run_sim_torch_diff_topologies(sim_len, J_C, spins, topologies, H, F_idx, avg_mag=None):
    """
    As _run_sim_torch(), but here topologies is a list of length n_realizations with a different topology for each realization
    """
    n_loops_per_top = np.array([len(top) for top in topologies])
    max_n_loops = np.max(n_loops_per_top)
    topologies = np.array(list(zip_longest(*topologies, fillvalue=0))).T
    new_spins = torch.empty_like(spins)
    all_realization_idx = torch.arange(len(topologies))[:, None]
    for t in range(sim_len):
        if avg_mag is not None:
            log_avg_mag_torch_diff_topologies(avg_mag, t, spins.detach().cpu().numpy(), topologies=topologies, n_loops_per_top=n_loops_per_top)
        new_spins.fill_(0)
        new_spins[:, :max_n_loops] += torch.matmul(J_C, spins[:, :max_n_loops, None]).squeeze(-1)
        new_spins += spins[all_realization_idx, F_idx]
        new_spins += H[t]  # TODO this works only for a scalar array
        spins[...] = torch.sign(new_spins)


@profile
def run_sim(exp_log: GloopNetExpLog, record_magnetization=True, record_full_state=False):
    """
    Setup relevant tensors, run simulations and save results
    :param exp_log: An GloopNetExpLog object with all relevant parameters for the simulation/s
    :param record_magnetization: To record and save the average magnetization of each realization at each time point
    :param record_full_state: To record and save the full spin state of each realization at each time point
    :return: None (Logs are saved to disk) # TODO perhaps add a return option?
    """
    # Generate required tensors
    if exp_log.topology is None:
        topologies = [gen_topology(n_spins=exp_log.n_spins) for _ in range(exp_log.n_realizations)]
        F_idx = get_diff_top_feeding_idx(topologies)
        J_sigmas = np.array([exp_log.gamma / np.sqrt(len(topology)) for topology in topologies])
        J_C = create_J_C_diff_topologies(topologies=topologies, J_C_sigmas=J_sigmas)
    else:
        F_idx = get_feeding_idx(exp_log.topology)
        J_sigma = exp_log.gamma / np.sqrt(len(exp_log.topology))
        J_C = create_J_C(J_C=exp_log.J_C, same_J_C=exp_log.same_J_C, J_C_sigma=J_sigma,
                         n_realizations=exp_log.n_realizations, topology=exp_log.topology)

    # Load into torch
    device = torch.device(exp_log.device)
    J_C = torch.as_tensor(J_C, dtype=TORCH_DTYPE, device=device)
    F_idx = torch.as_tensor(F_idx, dtype=torch.long, device=device)

    # Annnnd... run
    for (sim_len, H) in zip(exp_log.sim_len, exp_log.H):
        H = torch.as_tensor(H, dtype=TORCH_DTYPE, device=device)
        spins = create_spins(spins=exp_log.spins, same_init_spins=exp_log.same_init_spins, spins_p=exp_log.spins_p,
                             n_realizations=exp_log.n_realizations, n_spins=exp_log.n_spins)
        spins = torch.as_tensor(spins, dtype=TORCH_DTYPE, device=device)
        avg_mag = None if not record_magnetization else torch.zeros(size=(exp_log.n_realizations, sim_len), dtype=TORCH_DTYPE, device=device)
        full_state = None if not record_full_state else torch.zeros(size=(exp_log.n_realizations, sim_len, exp_log.n_spins), dtype=TORCH_DTYPE, device=device)

        if exp_log.topology is None:
            _run_sim_torch_diff_topologies(sim_len=sim_len, J_C=J_C, spins=spins, topologies=topologies, H=H,
                                           F_idx=F_idx, avg_mag=avg_mag)
        else:
            topology = torch.as_tensor(exp_log.topology, dtype=torch.long, device=device)
            _run_sim_torch(sim_len=sim_len, J_C=J_C, spins=spins, topology=topology, H=H,
                           F_idx=F_idx, avg_mag=avg_mag, full_state=full_state)
        if avg_mag is not None: np.save(f"{exp_log.dir}\\mag_results_sim_len={sim_len}", avg_mag.detach().cpu().numpy())
        if full_state is not None: np.save(f"{exp_log.dir}\\state_sim_len={sim_len}", full_state.cpu().numpy())

# @util.time_func
@jit(nopython=True, parallel=True)
def get_lag_times_jit(mag, stable_t, relaxation_t, n_std):
    """
    :param mag: Tensor of size (n_realizations, sim_len) with avg. of loop magnetizations (average per loop averaged over loops)
    :param stable_t: Tuple with start/end times of period where system is considered in ground state.
    :param relaxation_t: Tuple with start/end times in which to search for return to ground state (will report number of
            time steps until magnetization returns to mean+n_std*sigma, computed over stable_t)
    :param n_std: Return to ground state is considered when magnetization crosses the threshold mean+n_std*sigma
    :return: Return times to ground state for each realization
    """
    n_realizations, sim_len = mag.shape
    lag_times = - np.ones(n_realizations, dtype=np.int64)

    def get_lag_time(i, thresh):
        for t in prange(relaxation_t[0], relaxation_t[1]):
            if mag[i, t] < thresh:
                lag_times[i] = t - relaxation_t[0]  # TODO batch
                return

    for i in prange(n_realizations):
        mean = np.mean(mag[i, stable_t[0]:stable_t[1]])
        std = np.std(mag[i, stable_t[0]:stable_t[1]])
        thresh = mean + n_std * std
        get_lag_time(i, thresh)

    return lag_times


@util.time_func
def main():
    # TODO (top-priority) aut. divide big system (e.g. 2^18) and combine results
    # Usage example
    # n_spins = 2**12
    n_spins = 2**5
    topology = np.array(gen_topology(n_spins=n_spins))
    n_realizations = 2
    gamma = 1.5
    # J_C = create_J_C(J_C=None, same_J_C=False, J_C_sigma=gamma / np.sqrt(len(topology)), n_realizations=n_realizations, topology=topology)
    # T_Ws = [20, 40, 160, 640, 900, 1280, 2100, 3000]
    T_Ws = [10]
    sim_lens = [6000 + T_W for T_W in T_Ws]
    # J_C = create_J_C(J_C=None, same_J_C=True, J_C_sigma=gamma / np.sqrt(len(topology)), n_realizations=None, topology=topology)
    Hs = [np.zeros(shape=(sim_len, len(topology))) for sim_len in sim_lens]
    mag_field_strength = lambda n_spins: 0.8/(np.sqrt(n_spins/2**14))
    for H, T_W in zip(Hs, T_Ws): H[2000:2000+T_W, :] = mag_field_strength(n_spins)

    # The ExpLog object records all relevant information and is responsible for logging it before and during the experiment
    exp_log = GloopNetExpLog\
        (
            exp_name=f"2^{int(np.log2(n_spins))}_rand_half_loops_mag_spins_J",
            n_realizations=n_realizations,  # Scalar
            sim_len=sim_lens,  # Scalar or list for multiple runs
            H=Hs,  # 1D ndarray (sim_len), 2D ndarray (sim_len, n_loops) or list of arrays for multiple runs
            topology=topology,  # 1D array defining network topology. If None, different topologies are generated for each realization
            J_C=None,  # 2D array (n_loops, n_loops) of connecting spin interaction weight or None for randomizing
            same_J_C=False,  # (If J_C is None) Randomize J_C once or for each realization
            gamma=gamma,  # Only relevant if a new topology is generated per realization (common value 1.5)
            # J_C_sigma=None,  # Randomized J_C connections are drawn from N(0, sigma*I)
            n_spins=n_spins,
            spins=None,  # 1D array (n_spins) of initial values for spins or None for randomizing
            same_init_spins=False,  # (If spins is None) Randomize initial conditions once or for each realization
            spins_p=0.5,  # Randomized J_C connections are drawn from Bin(p) (if None, and same_init_spins=False, a new p~uni(0, 1) is drawn for each realization)
            save_log=True,  # Set to False for debugging (don't create a directory for the experiment)
            device="cuda:0"
        )
    run_sim(exp_log=exp_log)

    # Analyze results (get lag times)

    # for sim_len, T_W in zip(sim_lens, T_Ws):
    #     mag_results = np.load(f"{exp_log.dir}\\mag_results_sim_len={sim_len}.npy")
    #     lag_times = get_lag_times_jit(mag=mag_results, stable_t=(1000, 2000), relaxation_t=(2000+T_W, sim_len), n_std=0)
    #     np.save(f"{exp_log.dir}\\lag_times_results_sim_len={sim_len}", lag_times)


def run_tests():
    """
    RUN THIS AFTER EVERY CHANGE
    Compare run to standard (to make sure all is well after code changes)
    :return: None (Test results are printed)
    # TODO tests for multiple topologies
    """
    np.random.seed(0)

    n_realizations = 2
    n_spins = 2**14
    topology = gen_topology(n_spins=n_spins)
    gamma = 1.5

    T_Ws = [20]
    sim_lens = [60 + T_W for T_W in T_Ws]
    Hs = [np.zeros(shape=(sim_len, len(topology))) for sim_len in sim_lens]
    mag_field_strength = lambda n_spins: 0.8/(np.sqrt(n_spins/2**14))
    for H, T_W in zip(Hs, T_Ws): H[20:20+T_W, :] = mag_field_strength(n_spins)

    Test = namedtuple("Test", ["test_name", "same_J_C", "same_init_spins", "spins_p"])
    tests = \
        [
            Test(test_name="test_1", same_J_C=True, same_init_spins=False, spins_p=None),
            Test(test_name="test_2", same_J_C=True, same_init_spins=True, spins_p=0.5),
            Test(test_name="test_3", same_J_C=True, same_init_spins=False, spins_p=0.5),
            Test(test_name="test_4", same_J_C=False, same_init_spins=True, spins_p=0.5),
            Test(test_name="test_5", same_J_C=False, same_init_spins=False, spins_p=0.5),
        ]

    for test in tests:
        print(f"Running test with parameters: {test}")
        exp_log = GloopNetExpLog\
            (
                exp_name=test.test_name,
                n_realizations=n_realizations,
                sim_len=sim_lens,
                H=Hs,
                topology=topology,
                J_C=None,
                same_J_C=test.same_J_C,
                gamma=gamma,
                n_spins=n_spins,
                spins=None,
                same_init_spins=test.same_init_spins,
                spins_p=test.spins_p,
                save_log=True,
                device="cuda:0"
            )
        run_sim(exp_log=exp_log, record_full_state=True)
        time.sleep(1)
        test_results = \
            {"All spins": None,
             "Magnetization": None}

        all_spins_results = np.load(f"{exp_log.dir}\\state_sim_len={sim_lens[0]}.npy")
        compare_all_spins_results = np.load(f"experiments\\test_standard\\{test.test_name}\\state_sim_len={sim_lens[0]}.npy")
        test_results["All spins"] = np.all(all_spins_results == compare_all_spins_results)

        mag_results = np.load(f"{exp_log.dir}\\mag_results_sim_len={sim_lens[0]}.npy")
        compare_mag_results = np.load(f"experiments\\test_standard\\{test.test_name}\\mag_results_sim_len={sim_lens[0]}.npy")
        test_results["Magnetization"] = np.all(mag_results == compare_mag_results)

        for result in test_results:
            if not test_results[result]:
                util.print_red(f"{result} test failed with the following parameters:\n {test}")
        shutil.rmtree(exp_log.dir)


if __name__ == '__main__':
    # mag = np.load("experiments\\2021_09_17_13_22_38_2^12_rand_half_loops_mag_spins_J\\mag_results_sim_len=6010.npy")
    # lag = np.load("experiments\\2021_09_17_13_22_38_2^12_rand_half_loops_mag_spins_J\\lag_times_results_sim_len=6010.npy")
    # gloopNetAnalysis.plot_kill_curve(lag_times=lag)

    # main()
    run_tests()
