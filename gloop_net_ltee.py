import numpy as np
import pandas as pd
import altair as alt
import pickle
import matplotlib.pyplot as plt
from gloop_net_sim import *
from gloop_net_sim import _run_sim_torch, _run_sim_torch_only_lag_times, _run_sim_torch_only_lag_times_cyclical_shift
from gloop_net_evolution import create_J_C
import time
import os
from itertools import product
from abc import ABC, abstractmethod
from exp_log import LTEEExpLog
import util
import gloop_net_analysis
from line_profiler_pycharm import profile
import wandb

TORCH_DEVICE = torch.device("cuda:0")
TORCH_FLOAT_DTYPE = torch.float32

def _get_lag_hist(lag_times, max_t=None):
    if max_t is not None: lag_times[lag_times > max_t] = max_t
    rv = np.bincount(lag_times, minlength=max_t+1)
    return rv


def generate_mutants(n_mutations, N, g, f, t, T, new_genotype_idxs):
    N_cells_per_mutant = (N[g][t + 1] - N[g][t]) / (n_mutations + 1)  # TODO if mu>>1 this is a RV (LD distribution?)
    for new_g in new_genotype_idxs:
        N[new_g] = np.zeros(T, dtype=np.float32)
        f[new_g] = np.zeros(T, dtype=np.int32)
        N[g][t + 1] -= N_cells_per_mutant
        N[new_g][t + 1] += N_cells_per_mutant


def _get_T(genotypes, u_genotypes, lag_times, mu, K):
    N_tot, t = 0, 0
    max_t = np.max(lag_times)
    N = np.zeros(shape=len(u_genotypes))
    f = np.stack([_get_lag_hist(lag_times[genotypes==g], max_t=max_t) for g in u_genotypes])
    while N_tot<K:
        if t<f.shape[1]:
            N = N * np.exp(mu * (1 - N_tot / K)) + f[:, t]
        else:
            N = N * np.exp(mu * (1 - N_tot / K))
        N_tot, t = np.ceil(np.sum(N)), t+1
    return t


def ltee_growth(genotypes, lag_times, growth_rate, mutation_rate, carrying_capacity, all_genotypes, max_time=None):
    """
    Run a logistic growth simulating a LTEE round.
    :param genotypes: List of n genotype indices (non-unique) for each of the n initial cells
    :param lag_times: List of n recovery times (lag) for each of the n initial cells
    :param growth_rate: For the logistic growth equation, in units of one over RCCN-iteration.
    :param mutation_rate: Probability of mutation per division event  TODO may need a dictionary of g: mutation rate
    :param carrying_capacity: Logistic growth upper limit
    :param all_genotypes: A set with indices of all genotypes created thus far # TODO need only max generated genotype index
    :return:
    """
    mu, lam, K = growth_rate, mutation_rate, carrying_capacity
    u_genotypes = list(np.unique(genotypes))  # TODO may be sorted
    lag_times -= np.min(lag_times)  # No need to simulate the first steps where no genotype has yet recovered
    T = _get_T(genotypes, u_genotypes, lag_times, mu, K)  # The simulation runs twice, first time (without generating mutants etc) to estimate T (so arrays can be pre-allocated)
    if max_time is not None: T = min(T, max_time)
    N = {g: np.zeros(T, dtype=np.float32) for g in u_genotypes}  # Number of cells of each genotype at each time iteration
    f = {g: _get_lag_hist(lag_times[genotypes==g], max_t=T) for g in u_genotypes}  # Number of cells recovering at each time point
    for g in N: N[g][0] = f[g][0]
    parent_mutant_dict = {}
    for t in range(T - 1):
        N_tot = sum([N[g][t] for g in u_genotypes])
        new_mutants_iteration_t = []
        for g in u_genotypes:
            N[g][t+1] = N[g][t]*np.exp(mu * (1 - N_tot / K)) + f[g][t+1]  # Logistic growth (TODO exponential?)
            n_mutations = np.random.binomial(n=N[g][t+1] - N[g][t], p=lam)
            if n_mutations:
                new_genotype_idxs = list(range(max(all_genotypes) + 1, max(all_genotypes) + n_mutations + 1))
                all_genotypes.update(new_genotype_idxs)
                generate_mutants(n_mutations, N, g, f, t, T, new_genotype_idxs)
                parent_mutant_dict[g] = new_genotype_idxs if not g in parent_mutant_dict else parent_mutant_dict[g] + new_genotype_idxs
                new_mutants_iteration_t += new_genotype_idxs
        u_genotypes += new_mutants_iteration_t

    # So we report *all* cells and not only active ones
    for g in u_genotypes:
        N[g] += np.count_nonzero(genotypes==g)
    return N, parent_mutant_dict


def get_RCCN_lag_times_f(topology, H, C_idx, FC_idx, lag_start_time):
    def RCCN_lag_times(Js, spins):
        return _run_sim_torch_only_lag_times_cyclical_shift(Js, spins, topology, H, C_idx, FC_idx, lag_start_time).detach().cpu().numpy()
    return RCCN_lag_times


def sample_J_idxs(n_per_idx, sampled_population_size, J_dict):
    J_idxs = np.random.choice(np.repeat(list(n_per_idx.keys()), list(n_per_idx.values())), size=sampled_population_size)
    Js = torch.as_tensor(np.stack([J_dict[i] for i in J_idxs]), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    return J_idxs, Js



# @profile
def run_ltee(experiment_log):
    _ = experiment_log
    (Js, J_idxs, J_dict), spins, mut_params = _.gen_init_Js(), _.gen_init_spins(), _.gen_init_mut_params()
    all_genotypes = set(np.unique(J_idxs))
    RCCN_lag_times_f = get_RCCN_lag_times_f(_.topology, _.H, _.C_idx, _.FC_idx, _.lag_start_time)
    i = 0

    while not _.termination_cond(i):
        start_time, i = time.time(), i + 1
        spins = _.gen_init_spins()  # TODO regenerate spin vectors?
        lag_times = RCCN_lag_times_f(Js, spins)
        N, parent_mutant_dict = ltee_growth(J_idxs, lag_times, growth_rate=_.growth_rate, mutation_rate=mut_params, carrying_capacity=_.stationary_size, all_genotypes=all_genotypes, max_time=_.max_time)
        _.mutator(J_dict, parent_mutant_dict)
        J_idxs, Js = sample_J_idxs(n_per_idx={g: N[g][-1] for g in N}, sampled_population_size=_.bottleneck_size, J_dict=J_dict)

        _.logger.log_iteration(iteration=i, iteration_time=time.time() - start_time, N=N, mut_params=mut_params)

    _.logger.log_final(J_dict=J_dict, mutant_dict=_.mutator.mutant_dict)


class SingleJ:
    def __init__(self, J, J_pop_size):
        self.J_pop_size = J_pop_size
        self.J = J

    def __call__(self):
        Js = np.repeat(self.J[None, ...], self.J_pop_size, axis=0)
        J_idxs = np.zeros(self.J_pop_size, dtype=np.uint16)
        J_dict = {0: self.J}
        return torch.as_tensor(Js, dtype=TORCH_DTYPE, device=TORCH_DEVICE), J_idxs, J_dict

class SingleJAndMutants:
    def __init__(self, J, mutant_Js, J_pop_size):
        self.J_pop_size = J_pop_size
        self.J = J
        self.mutant_Js = mutant_Js

    def __call__(self):
        Js = np.repeat(self.J[None, ...], self.J_pop_size, axis=0)
        J_idxs = np.zeros(self.J_pop_size, dtype=np.uint16)
        J_dict = {0: self.J}
        for i in range(len(self.mutant_Js)):
            Js[i], J_idxs[i] = self.mutant_Js[i], i+1
            J_dict[i+1] = self.mutant_Js[i]
        return torch.as_tensor(Js, dtype=TORCH_DTYPE, device=TORCH_DEVICE), J_idxs, J_dict

def get_J_sigma(n_loops):
    return 1.5 / np.sqrt(n_loops)

class MixtureJ:
    def __init__(self, topology, N_Js, J_pop_size, rng_seed=None):
        self.topology = topology
        self.J_pop_size = J_pop_size
        self.N_Js = N_Js
        self.rng_seed = rng_seed

    def __call__(self):
        Js = create_J_C(J_C=None, same_J_C=False, J_C_sigma=get_J_sigma(len(self.topology)), n_realizations=self.N_Js, topology=self.topology, rng_seed=self.rng_seed)
        J_dict = {i: Js[i] for i in range(self.N_Js)}
        Js = np.repeat(Js, self.J_pop_size//self.N_Js, axis=0)
        J_idxs = np.repeat(np.arange(self.N_Js), self.J_pop_size//self.N_Js, axis=0)

        return torch.as_tensor(Js, dtype=TORCH_DTYPE, device=TORCH_DEVICE), J_idxs, J_dict


def mag_field_strength(n_spins):
    return 0.8 / (np.sqrt(n_spins / 2 ** 14))


def setup_stress(n_loops, n_spins, T_w, mag_loops=None):
    if mag_loops is None: mag_loops = np.arange(n_loops)
    sim_len = util.STABLE_T[1] + T_w + util.RELAXATION_TIME
    H = np.zeros(shape=(sim_len, n_loops))
    H[util.STABLE_T[1]:util.STABLE_T[1] + T_w, mag_loops] = mag_field_strength(n_spins)
    lag_start_time = util.STABLE_T[1] + T_w
    H = torch.as_tensor(H, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    return H, lag_start_time

def setup_init_spins(pop_size, n_spins):
    spins = create_spins(spins=None, same_init_spins=False, spins_p=0.5, n_realizations=pop_size, n_spins=n_spins)
    return torch.as_tensor(spins, dtype=TORCH_DTYPE, device=TORCH_DEVICE)


def plot_time_stack(all_N, avg_lag):
    total_genomes = list(set.union(*[set(N.keys()) for N in all_N]))
    # avg_lag = [avg_lag[g_i] for g_i in total_genomes]
    times = [len(next(iter(all_N[i].values()))) for i in range(len(all_N))]
    total_time = sum(times)
    # Dataset
    a = np.zeros(shape=(total_time, len(total_genomes)))
    s = np.cumsum([0] + times)
    s = zip(s[:-1], s[1:])
    for N in all_N:
        cur_s, cur_e = next(s)
        for g in total_genomes:
            if g in N:
                a[cur_s:cur_e, g] = N[g]
    df = pd.DataFrame.from_dict(
        {
            "ID": np.repeat(total_genomes, total_time),
            # "Avg_lag": np.repeat(avg_lag, total_time),
            "Time": np.tile(np.arange(total_time), len(total_genomes)),
            "Count": a.T.flatten()
        })

    df.to_csv("tst", index=False)


def competition_assay(Js, topology, T_w, init_size, final_size, growth_rate, lag_times=None):
    if lag_times is None:
        n_loops, n_spins = len(topology), np.sum(topology)
        H, lag_start_time = setup_stress(n_loops, n_spins, T_w=T_w, mag_loops=None)
        C_idx, _, FC_idx, _ = get_feeding_idx(topology, contiguous_C=False, return_only_F=False)
        C_idx, FC_idx = torch.as_tensor(C_idx, dtype=torch.long, device=TORCH_DEVICE), torch.as_tensor(FC_idx, dtype=torch.long, device=TORCH_DEVICE)
        topology = torch.as_tensor(topology, dtype=torch.long, device=TORCH_DEVICE)
        RCCN_lag_times_f = get_RCCN_lag_times_f(topology, H, C_idx, FC_idx, lag_start_time)
        genotypes = np.repeat(np.arange(len(Js)), init_size // len(Js), axis=0)
        spins = setup_init_spins(pop_size=init_size, n_spins=n_spins)
        Js = np.repeat(Js, init_size // len(Js), axis=0)
        Js = torch.as_tensor(Js, dtype=torch.float32, device=TORCH_DEVICE)
        lag_times = RCCN_lag_times_f(Js, spins)

    else:
        genotypes = np.repeat(np.arange(len(lag_times)), init_size // len(lag_times), axis=0)
        lag_times = np.concatenate(lag_times)

    N, _ = ltee_growth(genotypes=genotypes, lag_times=lag_times, growth_rate=growth_rate, mutation_rate=0, carrying_capacity=final_size, all_genotypes=set(np.arange(len(Js))))
    return N
    # return [N[i][-1] for i in N]

class Logger:
    def __init__(self, n_iterations, init_J=None, dir=None):
        self.dir = dir
        self.init_J = init_J
        self.all_N = [None] * n_iterations
        self.final_N = [None] * n_iterations


    def log_iteration(self, iteration, iteration_time, N, mut_params):

        print(f"Logging iteration {iteration}. Timed at {iteration_time:.3f} ({len(next(iter(N.values())))} rounds)")
        for k in N:
            print(f"strain {k}: {N[k][0]}-->{N[k][-1]}")
        self.all_N[iteration-1] = N
        self.final_N[iteration-1] = {k: v[-1] for k, v in N.items()}

    def log_final(self, J_dict, mutant_dict):
        self.J_dict = J_dict
        self.mutant_dict = mutant_dict
        self.save()

    def save(self):
        if self.init_J is not None: pickle.dump(self.init_J, open(f"{self.dir}\\init_J", 'wb'))
        # pickle.dump(self.all_N, open(f"{self.dir}\\all_N", 'wb'))
        pickle.dump(self.final_N, open(f"{self.dir}\\final_N", 'wb'))
        pickle.dump(self.mutant_dict, open(f"{self.dir}\\mutant_dict", 'wb'))

class SingleRowColumnMutation:
    def __init__(self, mut_sigma, n_mutated_nodes=1):
        self.mut_sigma = mut_sigma
        self.n_mutated_nodes = n_mutated_nodes
        self.mutant_dict = {}

    def __call__(self, J_dict, parent_mutant_dict):
        if not parent_mutant_dict: return
        n_loops = next(iter(J_dict.values())).shape[-1]
        for (parent_id, children) in parent_mutant_dict.items():
            for child_id in children:
                mutant_J = np.copy(J_dict[parent_id])
                mut_loop_i = np.random.choice(n_loops)
                mutated_connections = self.mut_sigma * np.random.randn(n_loops)
                row_column = np.random.choice(2)
                if row_column:
                    mutant_J[:, mut_loop_i] = mutated_connections
                else:
                    mutant_J[mut_loop_i, :] = mutated_connections
                mutant_J = ((mutant_J - np.mean(mutant_J)) / np.std(mutant_J)) * self.mut_sigma  # Standardize
                mutant_J[np.diag_indices(n_loops)] = 0
                J_dict[child_id] = mutant_J
                self.mutant_dict[child_id] = {"parent_id": parent_id,
                                              "mutated_node": mut_loop_i,
                                              "mutated_connections": mutated_connections,
                                              "row_col_mutation": ["row", "col"][row_column]}


def main():
    growth_rate = 0.1
    bottleneck_size = 2**10
    # for stationary_size in [2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17]:
    # for growth_rate in np.logspace(-3, 0, num=7):
    # for stationary_size in [2**11, 2**14, 2**17, 2**20]:
    for i in range(5):
        for max_time in [20, 50, 100, None]:
            print(f"{i=}, {max_time=}")
            bottleneck_size, stationary_size = 1000, 50_000
            n_iterations = 200
            # mutation_rate = 1 / (np.log2(stationary_size / bottleneck_size) * 50_000)
            mutation_rate = 1 / ((stationary_size - bottleneck_size) * 2)

            topology, init_J = np.load("topologies_and_Js\\topology_1.npy"), np.load("topologies_and_Js\\topology_1_J_1.npy")
            n_loops, n_spins = len(topology), np.sum(topology)
            T_w = 2000
            H, lag_start_time = setup_stress(n_loops, n_spins, T_w=T_w, mag_loops=None)

            logger = Logger(n_iterations)
            mutator = SingleRowColumnMutation(get_J_sigma(len(topology)))
            C_idx, _, FC_idx, _ = get_feeding_idx(topology, contiguous_C=False, return_only_F=False)
            C_idx, FC_idx = torch.as_tensor(C_idx, dtype=torch.long, device=TORCH_DEVICE), torch.as_tensor(FC_idx, dtype=torch.long, device=TORCH_DEVICE)
            MixtureJ(topology=topology, N_Js=bottleneck_size, J_pop_size=bottleneck_size, rng_seed=0)

            experiment_log = LTEEExpLog(
                exp_name=f"{bottleneck_size//1000}K_to_{stationary_size//1000}K_{T_w=}_mu={growth_rate:.3f}_{max_time=}_{i=}",
                topology=torch.as_tensor(topology, dtype=torch.long, device=TORCH_DEVICE),
                C_idx=C_idx,
                FC_idx=FC_idx,
                H=H,
                lag_start_time=lag_start_time,
                # gen_init_Js=MixtureJ(topology=topology, N_Js=bottleneck_size, J_pop_size=bottleneck_size, rng_seed=0),
                gen_init_Js=SingleJ(J=init_J, J_pop_size=bottleneck_size),
                gen_init_spins=lambda: setup_init_spins(pop_size=bottleneck_size, n_spins=n_spins),
                gen_init_mut_params=lambda: mutation_rate,
                mutator=mutator,
                termination_cond=lambda i: i >= n_iterations,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                stationary_size=stationary_size,
                max_time=max_time,
                logger=logger)

            run_ltee(experiment_log)

            # avg_lag = {g: get_avg_lag(topology=topology, J=logger.J_dict[g], T_w=T_w) for g in logger.J_dict}
            # plot_time_stack(logger.all_N, avg_lag=None)


if __name__ == '__main__':


    # wandb.init(project="my-test-project", entity="hrappeport")
    # wandb.init(
    #     # Set the project where this run will be logged
    #     project="tst1",
    #     entity="hrappeport",
    #     name=f"experiment_{run}",
    #     config={
    #         "learning_rate": 0.02,
    #         "architecture": "CNN",
    #         "dataset": "CIFAR-100",
    #         "epochs": 10,
    #     }
    #
    # wandb.log({"acc": acc, "loss": loss})
    #
    # wandb.finish()

    main()

