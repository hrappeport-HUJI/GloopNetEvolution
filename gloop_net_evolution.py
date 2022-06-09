import numpy as np
from gloop_net_sim import *
from gloop_net_sim import _run_sim_torch, _run_sim_torch_only_lag_hist
import time
import os
from itertools import product
from abc import ABC, abstractmethod
from exp_log import EvoGloopExpLog
import util
import gloop_net_analysis
from line_profiler_pycharm import profile


DEBUG = False


@profile
def run_evolution(exp_log, logger):
    population, mut_params = exp_log.gen_init_pop_f()
    pop_size = len(population)
    fitness = exp_log.fitness_f(population)
    i = 0
    logger.log_iteration(iteration=i, iteration_time=0, population=population, fitness=fitness, mut_params=mut_params)
    while not exp_log.termination_cond(fitness, i):
        time1 = time.time()
        i += 1

        # Obtain mutated children
        children, children_mut_params = exp_log.mutation_f(population, mut_params), exp_log.mut_params_mutation_f(mut_params)
        population, fitness, mut_params = np.concatenate([population, children]), np.concatenate([fitness, exp_log.fitness_f(children)]), np.concatenate([mut_params, children_mut_params])

        # Select
        selected_idx = exp_log.selection_f(population=population, fitness=fitness, new_population_size=pop_size)
        population, fitness, mut_params = population[selected_idx], fitness[selected_idx], mut_params[selected_idx]

        time2 = time.time()
        logger.log_iteration(iteration=i, iteration_time=time2 - time1, population=population, fitness=fitness, mut_params=mut_params)


class GenInitPop_f(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()


class GenJPop(GenInitPop_f):
    def __init__(self, J_pop_size, topology, init_mut_size):
        self.gamma = 1.5
        self.J_pop_size = J_pop_size
        self.topology = topology
        self.init_mut_size = init_mut_size

    def __call__(self):
        return create_J_C(J_C=None, same_J_C=False, J_C_sigma=self.gamma / np.sqrt(len(self.topology)),
                          n_realizations=self.J_pop_size, topology=self.topology), \
               self.init_mut_size*np.ones(self.J_pop_size)

class SpecificJ(GenInitPop_f):
    def __init__(self, J, J_pop_size, init_mut_params):
        self.J_pop_size = J_pop_size
        self.J = J
        self.init_mut_params = init_mut_params

    def __call__(self):
        return np.repeat(self.J[None, ...], self.J_pop_size, axis=0), \
               self.init_mut_params*np.ones(self.J_pop_size, dtype=np.int32)  # TODO this may not need to be an int


class Fitness_f(ABC):
    @abstractmethod
    def __call__(self, population):
        raise NotImplementedError()

    @staticmethod
    def get_fitness_p(fitness, rank=False, increasing=True, log=False):
        """
        Get a probability vector from a fitness vector
        :param fitness:
        :param epsilon: If nonzero, this is the minimal probability of any entry
        :param increasing: If true, increasing fitness equals increasing probability of being selected
        :param log: log-transform before normalizing
        :return:
        """
        if rank:
            fitness_p = np.argsort(fitness) if increasing else np.argsort(-fitness)
        else:
            fitness_p = fitness - np.min(fitness) if increasing else -fitness + np.max(fitness)
        if log:
            fitness_p = np.exp(fitness_p - np.max(fitness))
        fitness_p = fitness_p/np.sum(fitness_p)
        return fitness_p


class LagTime_F():
    def __init__(self, topology, n_realizations_per_J, T_w, sim_len, H, spins_p, n_spins,
                 stable_t=util.STABLE_T, relaxation_t=None, n_std=0):
        self.n_realizations_per_J = n_realizations_per_J
        self.spins_p = spins_p
        self.n_spins = n_spins
        self.n_std = n_std
        self.sim_len = sim_len
        self.stable_t = stable_t
        self.relaxation_t = relaxation_t if relaxation_t is not None else (stable_t[1] + T_w, self.sim_len)
        self.device = torch.device(util.DEVICE)

        self.F_idx = torch.as_tensor(get_feeding_idx(topology), dtype=torch.long, device=self.device)

        self.H = torch.as_tensor(H, dtype=TORCH_DTYPE, device=self.device)
        self.topology = torch.as_tensor(topology, dtype=torch.long, device=self.device)

    def __call__(self, population):
        avg_mag = torch.zeros(size=(self.n_realizations_per_J*len(population), self.sim_len), dtype=TORCH_DTYPE, device=self.device)
        spins = create_spins(spins=None, same_init_spins=False, spins_p=self.spins_p,
                             n_realizations=self.n_realizations_per_J*len(population), n_spins=self.n_spins)
        spins = torch.as_tensor(spins, dtype=TORCH_DTYPE, device=self.device)
        J_C = torch.as_tensor(population, dtype=TORCH_DTYPE, device=self.device)
        J_C = torch.repeat_interleave(J_C, self.n_realizations_per_J, dim=0)
        _run_sim_torch(sim_len=self.sim_len, J_C=J_C, spins=spins, topology=self.topology, H=self.H,
                       F_idx=self.F_idx, avg_mag=avg_mag, full_state=None)
        lag_times = get_lag_times_jit(mag=avg_mag.detach().cpu().numpy(), stable_t=self.stable_t, relaxation_t=self.relaxation_t, n_std=self.n_std)
        # lag_times = _run_sim_torch_only_lag_hist(J_C=J_C, spins=spins, topology=self.topology, H=self.H, F_idx=self.F_idx, lag_start_time=self.relaxation_t[0])
        return lag_times

class LagTimeHist:
    def __init__(self, topology, n_realizations_per_J, T_w, sim_len, H, spins_p, n_spins,
                 stable_t=util.STABLE_T, relaxation_t=None, n_std=0):
        self.n_realizations_per_J = n_realizations_per_J
        self.spins_p = spins_p
        self.n_spins = n_spins
        self.n_std = n_std
        self.sim_len = sim_len
        self.stable_t = stable_t
        self.relaxation_t = relaxation_t if relaxation_t is not None else (stable_t[1] + T_w, self.sim_len)
        self.device = torch.device(util.DEVICE)

        self.F_idx = torch.as_tensor(get_feeding_idx(topology), dtype=torch.long, device=self.device)
        self.H = torch.as_tensor(H, dtype=TORCH_DTYPE, device=self.device)
        self.topology = torch.as_tensor(topology, dtype=torch.long, device=self.device)

    def __call__(self, J):
        # avg_mag = torch.zeros(size=(self.n_realizations_per_J*len(population), self.sim_len), dtype=TORCH_DTYPE, device=self.device)
        spins = create_spins(spins=None, same_init_spins=False, spins_p=self.spins_p,
                             n_realizations=self.n_realizations_per_J, n_spins=self.n_spins)
        spins = torch.as_tensor(spins, dtype=TORCH_DTYPE, device=self.device)
        J_C = torch.as_tensor(J, dtype=TORCH_DTYPE, device=self.device)
        lag_times_hist = _run_sim_torch_only_lag_hist(J_C=J_C, spins=spins, topology=self.topology, H=self.H, F_idx=self.F_idx, lag_start_time=self.relaxation_t[0])
        return lag_times_hist

class AvgReturnTimeFitness(Fitness_f):  # TODO deprecate (move to mul)
    def __init__(self, topology, n_realizations_per_J, T_w, sim_len, H, spins_p, n_spins,
                 stable_t=util.STABLE_T, relaxation_t=None, n_std=0):
        self.n_realizations_per_J = n_realizations_per_J
        self.RT = LagTimeFitness(topology, n_realizations_per_J, T_w, sim_len, H, spins_p,
                                 n_spins, stable_t, relaxation_t, n_std)

    def __call__(self, population):
        lag_times = self.RT(population)
        mean_lag_time = np.zeros(len(population))
        for J_C_i, J_C in enumerate(population):
            mean_lag_time[J_C_i] = np.mean(lag_times[J_C_i*self.n_realizations_per_J:(J_C_i+1)*self.n_realizations_per_J])
        return mean_lag_time


class MulRunsAvgReturnTimeFitness(Fitness_f):
    def __init__(self, topology, n_realizations_per_J, T_ws, sim_lens, Hs, spins_p, n_spins,
                 stable_t=util.STABLE_T, relaxation_t=None, n_std=0):
        self.n_realizations_per_J = n_realizations_per_J
        self.RTs = [LagTimeFitness(topology, n_realizations_per_J, T_ws[i], sim_lens[i], Hs[i], spins_p,
                                   n_spins, stable_t, relaxation_t, n_std) for i in range(len(Hs))]

    def __call__(self, population):
        lag_times = np.zeros(shape=(len(self.RTs), self.n_realizations_per_J*len(population)))
        for i in range(len(self.RTs)):
            lag_times[i] = self.RTs[i](population)

        mean_lag_time = np.zeros(len(population))
        for J_C_i, J_C in enumerate(population):
            mean_lag_time[J_C_i] = np.mean(lag_times[:, J_C_i*self.n_realizations_per_J:(J_C_i+1)*self.n_realizations_per_J])
        return mean_lag_time


class ExpGrowthFromReturnFitness(Fitness_f):  # TODO deprecate (move to mul)
    def __init__(self, topology, n_realizations_per_J, T_w, sim_len, H, spins_p, n_spins,
                 stable_t=util.STABLE_T, relaxation_t=None, n_std=0, mu=1):
        self.mu=mu
        self.n_realizations_per_J = n_realizations_per_J
        self.RT = LagTimeFitness(topology, n_realizations_per_J, T_w, sim_len, H, spins_p,
                                 n_spins, stable_t, relaxation_t, n_std)

    @staticmethod
    def get_N(mu, max_t=None, lag_times=None, lag_times_hist=None):
        if lag_times_hist is None:
            if max_t is None: max_t = int(5/mu) # TODO 5?
            lag_times += 1  # TODO
            lag_times[lag_times>max_t] = max_t
            lag_times_hist = np.bincount(lag_times, minlength=max_t+1)

        N=0
        for bin in lag_times_hist:
            N = np.exp(mu)*N + bin
        #
        # N = lag_times_hist[0]
        # for t in range(1, max_t+1):
        #     N = np.exp(mu)*N + lag_times_hist[t]
        return N

    def __call__(self, population):
        lag_times = self.RT(population)
        N = np.zeros(len(population))
        for J_C_i, J_C in enumerate(population):  # TODO this can be batched
            N[J_C_i] = ExpGrowthFromReturnFitness.get_N(lag_times=lag_times[J_C_i*self.n_realizations_per_J:(J_C_i+1)*self.n_realizations_per_J], mu=self.mu)
        return N

# class MulExpGrowthFromReturnFitness(Fitness_f):
#     def __init__(self, topology, n_realizations_per_J, T_ws, sim_lens, Hs, spins_p, n_spins,
#                  stable_t=util.STABLE_T, relaxation_t=None, n_std=0, mu=0.005):
#         self.mu=mu
#         self.n_realizations_per_J = n_realizations_per_J
#         self.RTs = [LagTimeFitness(topology, n_realizations_per_J, T_ws[i], sim_lens[i], Hs[i], spins_p,
#                                    n_spins, stable_t, relaxation_t, n_std) for i in range(len(Hs))]

    # @profile
    # def __call__(self, population):
    #     lag_times = np.zeros(shape=(len(self.RTs), self.n_realizations_per_J*len(population)), dtype=np.int64)
    #     for i in range(len(self.RTs)):
    #         lag_times[i] = self.RTs[i](population)
    #
    #     N = np.zeros(len(population))
    #     for J_C_i, J_C in enumerate(population):
    #         N[J_C_i] = np.mean(
    #             [ExpGrowthFromReturnFitness.get_N(lag_times[j, J_C_i*self.n_realizations_per_J:(J_C_i+1)*self.n_realizations_per_J], mu=self.mu) / self.n_realizations_per_J
    #              for j in range(len(self.RTs))])
    #     return N

class MulExpGrowthFromReturnFitness(Fitness_f):
    def __init__(self, topology, n_realizations_per_J, T_ws, sim_lens, Hs, spins_p, n_spins,
                 stable_t=util.STABLE_T, relaxation_t=None, n_std=0, mu=0.005):
        self.mu = mu
        self.n_realizations_per_J = n_realizations_per_J
        self.RTs = [LagTimeHist(topology, n_realizations_per_J, T_ws[i], sim_lens[i], Hs[i], spins_p,
                                   n_spins, stable_t, relaxation_t, n_std) for i in range(len(Hs))]
    @profile
    def __call__(self, population):
        get_N = ExpGrowthFromReturnFitness.get_N
        return np.array([np.mean([get_N(mu=self.mu, lag_times_hist=RT(J)) for RT in self.RTs]) for J in population])

class Selection_f(ABC):
    @abstractmethod
    def __call__(self, population, fitness, new_population_size):
        raise NotImplementedError()


class PropSelectionWithElitism(Selection_f):
    def __init__(self, n_elite, increasing, rank=False):
        self.increasing = increasing
        self.n_elite = n_elite
        self.rank = rank

    def __call__(self, population, fitness, new_population_size):
        selected_idx = np.zeros(new_population_size, dtype=np.int64)
        arg_sort_fitness = np.argsort(fitness) if not self.increasing else np.argsort(-fitness)
        selected_idx[:self.n_elite] = arg_sort_fitness[:self.n_elite]
        fitness_p = Fitness_f.get_fitness_p(fitness=fitness, rank=self.rank, increasing=self.increasing)
        selected_idx[self.n_elite:] = np.random.choice(len(population), replace=True, p=fitness_p, size=new_population_size-self.n_elite)
        return selected_idx

class TournamentSelection(Selection_f):
    def __init__(self, k, n_elite, increasing):
        self.k = k
        self.increasing = increasing
        self.n_elite = n_elite

    def __call__(self, population, fitness, new_population_size):
        selected_idx = np.zeros(new_population_size, dtype=np.int64)
        arg_sort_fitness = np.argsort(fitness) if not self.increasing else np.argsort(-fitness)
        selected_idx[:self.n_elite] = arg_sort_fitness[:self.n_elite]
        tournament_winners = np.min(np.stack([np.random.choice(len(population), replace=False, size=self.k) for _ in range(new_population_size-self.n_elite)]), axis=1)
        selected_idx[self.n_elite:] = arg_sort_fitness[tournament_winners]
        return selected_idx


class Mutation_f(ABC):
    @abstractmethod
    def __call__(self, population, mut_params):
        # Return a new generation (not necesarily of the same size) composed from the current one by mutating each member of the population
        raise NotImplementedError()


class GaussianMutation(Mutation_f):
    # def __init__(self, n_mutated_nodes, n_children_per_parent):
    #     self.n_mutated_nodes = n_mutated_nodes
    #     self.n_children_per_parent = n_children_per_parent

    # def __call__(self, population, mut_sigmas):
    #     # Note! Diagonal is not zeroed here
    #     n_loops = population.shape[-1]
    #     new_pop_size = len(population)*self.n_children_per_parent
    #     if self.n_mutated_nodes == n_loops**2:
    #         mut_sigmas = np.repeat(mut_sigmas, self.n_children_per_parent, axis=0)[:, None, None]  # dim expansion is for matmul compatibility
    #         epsilon = mut_sigmas * np.random.randn(new_pop_size*n_loops*n_loops).reshape(new_pop_size, n_loops, n_loops)
    #     else:
    #         epsilon = np.zeros(shape=(new_pop_size, n_loops, n_loops))  # Perturbation vector
    #         for parent_i, parent in enumerate(population): # TODO vectorize
    #             mut_rows = np.random.choice(n_loops, replace=True, size=self.n_mutated_nodes)
    #             mut_cols = np.random.choice(n_loops, replace=True, size=self.n_mutated_nodes)
    #             epsilon[parent_i, mut_rows, mut_cols] = mut_sigmas[parent_i] * np.random.randn(self.n_mutated_nodes)
    #     population = np.repeat(population, self.n_children_per_parent, axis=0) + epsilon
    #     return population

    def __init__(self, mut_sigma, n_children_per_parent):
        self.mut_sigma = mut_sigma
        self.n_children_per_parent = n_children_per_parent

    def __call__(self, population, n_mutated_nodes_per_J):
        # TODO should n_mutated_nodes be the mean of a distribution?
        # Note! Diagonal is not zeroed here (zeroed in GaussianMutationWithStandardization)
        n_loops = population.shape[-1]
        new_pop_size = len(population)*self.n_children_per_parent
        population = np.repeat(population, self.n_children_per_parent, axis=0)
        n_mutated_nodes_per_J = np.repeat(n_mutated_nodes_per_J, self.n_children_per_parent, axis=0)
        n_mutated_nodes = np.sum(n_mutated_nodes_per_J)
        epsilon = self.mut_sigma * np.random.randn(n_mutated_nodes, n_loops)
        mutated_loop_idx = np.random.choice(n_loops, size=n_mutated_nodes)  # Indices for each mutated node
        in_out_mutation = np.random.choice(2, size=n_mutated_nodes)  # mutate column or row (incoming or outgoing connections) for each mutation

        mut_i = 0
        for J_i, J in enumerate(population):
            for J_i_mut_j in range(n_mutated_nodes_per_J[J_i]):
                if in_out_mutation[mut_i] == 0:  # Mutated column (outgoing connections for the selected node)
                    J[:, mutated_loop_idx[mut_i]] = epsilon[mut_i]
                else:
                    J[mutated_loop_idx[mut_i], :] = epsilon[mut_i]
                mut_i += 1
        return population


class GaussianMutationWithStandardization(GaussianMutation):
    def __init__(self, mean, std, **qwargs):
        self.mean = mean
        self.std = std
        super().__init__(**qwargs)

    def standardize(self, population):
        population = (population - np.mean(population, axis=(1, 2))[:, None, None]) / np.std(population, axis=(1, 2))[:, None, None]
        population = (population + self.mean) * self.std
        pop_size, n_loops, _ = population.shape
        for i in range(pop_size): population[i][np.diag_indices(n_loops)] = 0  # Zero Diagonal
        return population

    def __call__(self, population, mut_params):
        mutated_population = super().__call__(population, mut_params)
        standardized_population = self.standardize(mutated_population)
        return standardized_population

class MutParamsMutation_f(ABC):
    @abstractmethod
    def __call__(self, mut_params):
        raise NotImplementedError()

class LogNormalMutParamsMutation_f(MutParamsMutation_f):
    def __init__(self, tau, n_children_per_parent, min_mut, max_mut):
        self.tau = tau
        self.min_mut = min_mut
        self.max_mut = max_mut
        self.n_children_per_parent = n_children_per_parent

    def __call__(self, mut_params):
        return np.clip(a=np.repeat(mut_params, self.n_children_per_parent, axis=0) *
                      np.exp(self.tau* np.random.randn(len(mut_params)*self.n_children_per_parent)),
                       a_min=self.min_mut, a_max=self.max_mut)

class ExpMutParamsMutation_f(MutParamsMutation_f):
    def __init__(self, increase_factor, increase_prob, min_mut, max_mut, n_children_per_parent):
        self.increase_factor = increase_factor
        self.increase_prob = increase_prob
        self.min_mut = min_mut
        self.max_mut = max_mut
        self.n_children_per_parent = n_children_per_parent

    def __call__(self, mut_params):
        # TODO allow for no mutation?
        return np.repeat(mut_params, self.n_children_per_parent, axis=0)


        increase = np.random.choice(a=[True, False], size=len(mut_params), p=self.increase_prob)
        mut_params[increase] = np.ceil(mut_params[increase]*self.increase_factor)
        mut_params[~increase] = np.ceil(mut_params[increase]/self.increase_factor)
        return np.clip(mut_params, a_min=self.min_mut, a_max=self.max_mut)


class TerminationCondition(ABC):
    @abstractmethod
    def __call__(self, fitness, iteration):
        raise NotImplementedError()

class MaxIterationTermination(TerminationCondition):
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    def __call__(self, fitness, iteration):
        return iteration >= self.max_iterations


class Logger(ABC):
    @abstractmethod
    def log_iteration(self, iteration, iteration_time, population, fitness, mut_params):
        raise NotImplementedError()

class IterationLogger(Logger):
    def __init__(self, log_freq, save_freq, increasing_fitness, verbose, n_iterations, population_size, save_dir, exp_log):
        self.save_dir = save_dir
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.increasing_fitness = increasing_fitness
        self.verbose = verbose
        self.n_iterations = n_iterations
        self.n_logs = len(range(0, self.n_iterations+1, self.log_freq))
        self.fitness_traj = np.zeros(shape=(self.n_logs, population_size))
        self.mut_params_traj = np.zeros(shape=(self.n_logs, population_size))
        self.best_sol = None
        self.exp_log = exp_log

    def save_iteration_cond(self, iteration):
        return iteration % self.save_freq == 0 or iteration == self.n_iterations

    def log_iteration(self, iteration, iteration_time, population, fitness, mut_params):
        if iteration == 0: self.best_sol = np.zeros(shape=(self.n_logs, *population.shape[1:]))

        print(f"Evolution iteration {iteration} completed. Timed at {iteration_time:.3f} s")
        if iteration%self.log_freq == 0:

            mean_fitness = np.mean(fitness)
            best_fitness = np.max(fitness) if self.increasing_fitness else np.min(fitness)
            # J_avg = np.mean(population, axis=(1, 2))
            if self.verbose:
                print(f"Average/Best fitness = {mean_fitness:.3f}/{best_fitness:.3f}")

            self.fitness_traj[iteration//self.log_freq] = fitness
            self.mut_params_traj[iteration//self.log_freq] = mut_params
            self.best_sol[iteration//self.log_freq] = population[np.argmax(fitness)] if self.increasing_fitness else population[np.argmin(fitness)]

        if self.save_iteration_cond(iteration):
            np.save(f"{self.save_dir}\\best_solutions.npy", self.best_sol)
            np.save(f"{self.save_dir}\\fitness_traj.npy", self.fitness_traj)
            np.save(f"{self.save_dir}\\mut_params_traj.npy", self.mut_params_traj)

        if iteration == self.n_iterations:
            np.save(f"{self.save_dir}\\final_population.npy", population)

def mag_field_strength(n_spins):
    return 0.8/(np.sqrt(n_spins/2**14))

# 5: 15
# 50:
# 500

def main():
    # TODO integralag selection? What mu?
    # TODO how many realizations per J are required for an accurate estimate of the mean? What about the integralag?
    # TODO recombination?
    # TODO effect of n_mutated_entries
    pop_size = 25
    n_realizations_per_J = 100  # On which to compute lag statistics
    # pop_size = 5
    # n_realizations_per_J = 10  # On which to compute lag statistics

    n_spins = 2**14 if not DEBUG else 2**8
    spins_p = 0.5
    topology, J = np.load("topologies_and_Js\\topology_1.npy"), np.load("topologies_and_Js\\topology_1_J_1.npy")
    get_J_sigma = lambda n_loops: 1.5 / np.sqrt(n_loops)
    if DEBUG:
        topology = np.array(gen_topology(n_spins=n_spins))
        J = create_J_C(J_C=None, same_J_C=True, J_C_sigma=get_J_sigma(len(topology)), n_realizations=None, topology=topology)

    # T_w = 2000 if not DEBUG else 20  # TODO
    # sim_len = STABLE_T[1] + T_w + RELAXATION_TIME
    # mag_idx = np.random.choice(np.argsort(topology)[:2*np.count_nonzero(topology==1)], n_mag_loops, replace=False) if n_mag_loops<2*np.count_nonzero(topology==1) else np.argsort(topology)[:n_mag_loops]
    # Evolution parameters
    n_elite = 3
    n_evolution_iterations = 99
    n_children_per_parent = 2

    n_mag_idx = len(topology) // 4
    # for fitness_f_type_str, T_ws in product(["avg", "exp_growth"], [[2000], [100]]):
    for round_i in range(1):
        print(f"{round_i=}")
        # for T_ws, mu in product([[2000], [100]], [0.001, 0.01, 0.1]):
        for T_ws, mu in product([[100]], [0.001, 0.01, 0.1]):
            fitness_f_type_str = "exp_growth"
            time1 = time.time()
            # try:
            mag_loops = np.arange(n_mag_idx)
            sim_lens = [util.STABLE_T[1] + T_w + util.RELAXATION_TIME for T_w in T_ws]
            Hs = [np.zeros(shape=(sim_len, len(topology))) for sim_len in sim_lens]

            for H, T_w in zip(Hs, T_ws): H[util.STABLE_T[1]:util.STABLE_T[1]+T_w, mag_loops] = mag_field_strength(n_spins)
            # for H, T_w in zip(Hs, T_ws): H[util.STABLE_T[1]:util.STABLE_T[1]+T_w, mag_loops] = np.repeat(np.linspace(0, mag_field_strength(n_spins), T_w)[:, None], n_mag_idx, axis=1)

            fitness_f_type = {"avg": MulRunsAvgReturnTimeFitness,
                              "exp_growth": MulExpGrowthFromReturnFitness}[fitness_f_type_str]
            increasing_fitness=fitness_f_type_str=="exp_growth "

            exp_log = EvoGloopExpLog(
                exp_name=f"{fitness_f_type_str}_fitness_mu={mu}_T_ws={T_ws}",
                # gen_init_pop_f=GenJPop(J_pop_size=pop_size, topology=topology, init_mut_size=5*1.5 / np.sqrt(len(topology))),
                gen_init_pop_f=SpecificJ(J=J, J_pop_size=pop_size, init_mut_params=1),
                fitness_f=fitness_f_type(topology=topology, n_realizations_per_J=n_realizations_per_J, T_ws=T_ws,
                                         sim_lens=sim_lens, Hs=Hs, spins_p=spins_p, n_spins=n_spins, mu=mu),
                selection_f=PropSelectionWithElitism(n_elite=n_elite, rank=False, increasing=increasing_fitness),
                # selection_f=TournamentSelection(k=tournament_k, n_elite=n_elite, increasing=increasing_fitness),
                mutation_f=GaussianMutationWithStandardization(mean=0, std=get_J_sigma(len(topology)),
                                                               mut_sigma=get_J_sigma(len(topology)),
                                                               n_children_per_parent=n_children_per_parent),
                mut_params_mutation_f=ExpMutParamsMutation_f(increase_factor=1.5, increase_prob=0.5, min_mut=0, max_mut=len(topology),
                                                             n_children_per_parent=n_children_per_parent),
                termination_cond=MaxIterationTermination(max_iterations=n_evolution_iterations),
                save_log=True,
                device="cuda:0")

            np.save(f"{exp_log.dir}\\topology.npy", topology)
            np.save(f"{exp_log.dir}\\mag_loops.npy", mag_loops)
            log_freq = 5  # TODO specific iterations (need better resolution in the first iterations)
            logger = IterationLogger(log_freq=log_freq, increasing_fitness=increasing_fitness, save_freq=n_evolution_iterations,
                                     verbose=True, n_iterations=n_evolution_iterations,
                                     population_size=pop_size, save_dir=exp_log.dir, exp_log=exp_log)
            run_evolution(exp_log=exp_log, logger=logger)

            # for i in range(n_evolution_iterations+1):
            #     gloop_net_analysis.generate_lag_times_for_best_solution_i(exp_name=exp_log.exp_name, evo_i=i, log_freq=log_freq)
            time2 = time.time()
            print(f"Exp {exp_log.exp_name} completed. Timed at {(time2 - time1):.3f} s")

        # except Exception as e:
            #     print(f"{exp_log.exp_name}\n{e}")

#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
    # for exp_name in ["2021_09_12_18_31_53_gradual_stress_avg_fitness_T_ws=[100]",
    #                  "2021_09_12_21_00_28_gradual_stress_exp_growth_fitness_T_ws=[2000]",
    #                  "2021_09_12_23_53_35_gradual_stress_exp_growth_fitness_T_ws=[100]"]:
    #     gloop_net_analysis.gen_kill_curve_gif(exp_name=exp_name, lag=5)

    # for i, exp_name in enumerate(os.listdir(f"experiments\\EvoGloop")):
    #     if i>=30:
    #         # print(exp_name)
    #         gloopNetAnalysis.gen_kill_curve_gif(exp_name=exp_name, lag=4)

