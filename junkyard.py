#
#
# def run_sim_sequential(sim_params):
#     s = parse_sim_params(sim_params, is_torch=False)
#     for realization_i in range(s.n_realizations):
#         spins = np.random.choice([-1, 1], size=s.n_spins)  # TODO binary
#         for t in range(s.sim_len):
#             new_spins = np.zeros_like(spins)
#             new_spins[s.R_idx] = spins[s.FR_idx]
#             cur_H = s.H if s.H_times[t] else 0
#             new_spins[s.C_idx] = np.sign(s.J_C @ spins[s.C_idx] + spins[s.FC_idx] + cur_H)
#             spins = new_spins
#
#
# def run_sim_vectorized(sim_params):
#     s = parse_sim_params(sim_params, is_torch=False)
#     J_C_b = np.broadcast_to(s.J_C, shape=(s.n_realizations, *s.J_C.shape))
#     spins = np.random.choice([-1, 1], size=(s.n_realizations, s.n_spins))  # TODO binary
#     for t in range(s.sim_len):
#         new_spins = np.zeros_like(spins)
#         new_spins[:, s.R_idx] = spins[:, s.FR_idx]
#         cur_H = s.H if s.H_times[t] else 0
#         new_spins[:, s.C_idx] = np.sign((J_C_b @ spins[:, s.C_idx][:, :, None]).squeeze() + spins[:, s.FC_idx] + cur_H)
#         spins = new_spins
#
#
# def run_sim_torch_sequential(sim_params):
#     s = parse_sim_params(sim_params, is_torch=True)
#     for _ in range(s.n_realizations):
#         spins = np.random.choice([-1, 1], size=(s.n_spins))  # TODO binary
#         spins = torch.tensor(spins, device=s.device)
#         for t in range(s.sim_len):
#             new_spins = torch.zeros_like(spins)
#             new_spins[s.R_idx] = spins[s.FR_idx]
#             cur_H = s.H if s.H_times[t] else 0
#             new_spins[s.C_idx] = torch.sign(s.J_C @ spins[s.C_idx].double() + spins[s.FC_idx] + cur_H).int()
#             spins = new_spins
#
#
# @jit(nopython=True, parallel=True)
# def run_sim_numba(
#         n_realizations,
#         sim_len,
#         J_C,
#         loop_sizes,
#         n_loops,
#         n_spins,
#         J_sigma,
#         R_idx,
#         FR_idx,
#         C_idx,
#         FC_idx,
#         H):
#     results = np.zeros(shape=(n_realizations, sim_len))
#     if J_C is not None:  # Single J matrix, randomized initial conditions
#         J_C_b = np.broadcast_to(J_C, shape=(n_realizations, *J_C.shape))
#         # TODO Adi's method with a different mean for each realization
#         spins = np.random.choice(np.array(-1, 1), size=(n_realizations, n_spins))
#     else:  # Randomized J matrix, identical initial conditions
#         J_C_b = J_sigma * np.random.randn(n_realizations, n_loops, n_loops)
#         spins = np.random.choice(np.array([-1, 1]), size=n_spins).astype(np.float64)
#         spins = np.repeat(np.expand_dims(spins, 0), n_realizations).reshape(n_realizations, n_spins).T
#
#     new_spins = np.zeros_like(spins)
#     for t in prange(sim_len):
#         for i in prange(n_realizations):
#             # log_results_np(results, t, spins, C_idx=C_idx, loop_sizes=loop_sizes)
#             for loop_c_i, loop_c in enumerate(C_idx):
#                 results[i, t] += np.mean(spins[i, loop_c:loop_c + loop_sizes[loop_c_i]])
#
#             new_spins[i][R_idx] = spins[i][FR_idx]
#             new_spins[i][C_idx] = np.sign(J_C_b[i] @ spins[i][C_idx] + spins[i][FC_idx] + H[t])
#             spins = new_spins
#     results /= n_loops
#     return results
#
#
# def parse_sim_params(sim_params, is_torch):
#     loop_sizes = sim_params["loop_sizes"]
#     J_C = sim_params["J_C"]
#
#     n_loops = len(loop_sizes)
#     n_spins = np.sum(loop_sizes)
#     C_idx = np.zeros_like(loop_sizes)  # C spins (communicating)
#     C_idx[1:] = np.cumsum(loop_sizes[:-1])
#     mask = np.zeros(n_spins, dtype=bool)
#     mask[C_idx] = True
#     R_idx = np.arange(n_spins)[~mask]  # R spins (Rest - Non communicating)
#     FC_idx = np.array([cs_ls - 1 for cs_ls in np.cumsum(loop_sizes)])  # spins feeding into C spins
#     FR_idx = R_idx - 1  # spins feeding into R spins
#
#     if is_torch:
#         R_idx = torch.tensor(R_idx, dtype=torch.long, device=sim_params["device"])
#         FR_idx = torch.tensor(FR_idx, dtype=torch.long, device=sim_params["device"])
#         C_idx = torch.tensor(C_idx, dtype=torch.long, device=sim_params["device"])
#         FC_idx = torch.tensor(FC_idx, dtype=torch.long, device=sim_params["device"])
#         J_C = torch.tensor(J_C, device=sim_params["device"]) if J_C is not None else None
#         sim_params["J_C"] = J_C
#
#     sim_params["n_loops"] = n_loops
#     sim_params["n_spins"] = n_spins
#     sim_params["C_idx"] = C_idx
#     sim_params["R_idx"] = R_idx
#     sim_params["FC_idx"] = FC_idx
#     sim_params["FR_idx"] = FR_idx
#     s = SimParams()
#     s.__dict__ = sim_params
#     return s
#
#
# # def log_results_np(results, t, spins, C_idx, loop_sizes):
# #     for loop_c_i, loop_c in enumerate(C_idx):
# #         results[:, t] += np.mean(spins[:, loop_c:loop_c + loop_sizes[loop_c_i]], dim=1)
# #     results[:, t] /= len(loop_sizes)
#
#
#
# # def run_sim_jax(
# #         n_realizations,
# #         sim_len,
# #         J_C,
# #         loop_sizes,
# #         n_loops,
# #         n_spins,
# #         J_sigma,
# #         R_idx,
# #         FR_idx,
# #         C_idx,
# #         FC_idx,
# #         H):
# #     results = torch.zeros(size=(n_realizations, sim_len), device=device, dtype=torch.double)
# #     if J_C is not None:  # Single J matrix, randomized initial conditions
# #         J_C_b = J_C.expand(n_realizations, *J_C.shape)
# #         # TODO Adi's method with a different mean for each realization
# #         spins = np.random.choice([-1, 1], size=(n_realizations, n_spins))
# #         spins = torch.tensor(spins, device=device, dtype=torch.double)
# #     else:  # Randomized J matrix, identical initial conditions
# #         J_C_b = J_sigma * torch.randn(n_realizations, n_loops, n_loops, dtype=torch.double, device=device)  # TODO CUDA out of memory can occur if too many realizations
# #         spins = np.random.choice([-1, 1], size=n_spins)
# #         spins = torch.tensor(spins, device=device, dtype=torch.double).expand(n_realizations, n_spins)
# #
# #     new_spins = torch.zeros_like(spins)
# #     for t in range(sim_len):
# #         log_results(results, t, spins, C_idx=C_idx, loop_sizes=loop_sizes)
# #         new_spins[:, R_idx] = spins[:, FR_idx]
# #         new_spins[:, C_idx] = torch.sign((J_C_b @ (spins[:, C_idx][:, :, None])).squeeze() + spins[:, FC_idx] + H[t])
# #         spins = new_spins
# #     return results
#
# def get_return_times(results, stable_t_idx, mag_end_t, n_std):
#     """
#     :param results: Tensor of size (n_realizations, sim_len) with
#     sum of loop avg magnetizations ()
#     :param stable_t_idx: Tuple with start/end times after transient period and before magnetic field is applied.
#      System is considered to be in ground state
#     :param mag_end_t: Last time point magnetic field is applied
#     :param n_std: Size of flucuation beneath which is considered return to ground state
#     :return: Return times to ground state for each realization
#     """
#     n_realizations, sim_len = results.shape
#     return_times = np.zeros(n_realizations)
#
#     # Mode A - Compute ground state w.r.t pre T_w time (after transient)
#     mean = np.mean(results[:, stable_t_idx[0]:stable_t_idx[1]], axis=1)
#     std = np.std(results[:, stable_t_idx[0]:stable_t_idx[1]], axis=1)
#
#     # Mode B - Compute ground state w.r.t post T_w time
#     # stable_t = stable_t_idx[1] - stable_t_idx[0]
#     # mean = np.mean(results[:, -stable_t:], axis=1)
#     # std = np.std(results[:, -stable_t:], axis=1)
#
#     thresh = mean + n_std * std
#     for i in range(n_realizations):
#         return_times[i] = np.argmax(results[i, mag_end_t:] < thresh[i])
#     return return_times


# J_C = np.array([[0, 1.5, 3, 1],
#                 [1.1, 1, 4, 1.2],
#                 [1.1, 1, 0, 1.1],
#                 [2, 3.1, 0.5, 0]])
# J_C = np.load("J_reduced.npy")


# def get_idx_arrays(loop_sizes):
#     n_spins = np.sum(loop_sizes)
#     C_idx = np.zeros_like(loop_sizes)  # C spins (communicating)
#     C_idx[1:] = np.cumsum(loop_sizes[:-1])
#     mask = np.zeros(n_spins, dtype=np.bool_)
#     mask[C_idx] = True
#     R_idx = np.arange(n_spins)[~mask]  # R spins (Rest - Non communicating)
#
#     # FC_idx = np.array([cs_ls - 1 for cs_ls in np.cumsum(loop_sizes)])  # spins feeding into C spins
#     # FR_idx = R_idx - 1  # spins feeding into R spins
#     # TODO yuck yuck yuck this is only for the sanity check with Shaked's code
#     FC_idx = np.array([C_idx[i] + 1 if loop_sizes[i] >1 else C_idx[i] for i in range(len(loop_sizes))])  # spins feeding into C spins
#     FR_idx = np.zeros_like(R_idx)
#     idx = np.zeros(len(loop_sizes)+1, dtype=np.int32)  # C spins (communicating)
#     idx[1:] = np.cumsum(loop_sizes)
#     x = np.arange(n_spins)
#     cur_i=0
#     for i, loop_size in enumerate(loop_sizes):
#         if loop_size>1:
#             cur = x[idx[i]+1:idx[i+1]]+1
#             cur[-1] = C_idx[i]
#             FR_idx[cur_i:cur_i+len(cur)] = cur
#             cur_i+=len(cur)
#
#     return C_idx, R_idx, FC_idx, FR_idx


#
# perm_idx = torch.zeros(spins.shape[-1], dtype=torch.long)
# perm_idx[C_idx] = torch.arange(len(C_idx))
# perm_idx[R_idx] = torch.arange(len(C_idx), len(C_idx)+len(R_idx))
# FC_idx = perm_idx[FC_idx]
# FR_idx = perm_idx[FR_idx]
# anti_perm_idx = torch.empty_like(perm_idx)
# anti_perm_idx[perm_idx] = torch.arange(len(perm_idx))



# n_realizations = 3
# T = 1
# device = torch.device("cuda:0")
# avg_mag_1 = torch.zeros(size=(n_realizations, T), device=device)
# avg_mag_2 = np.zeros(shape=(n_realizations, T))
# avg_mag_3 = torch.zeros(size=(n_realizations, T), device=device)
# topology = torch.tensor([3, 1, 4, 2], device=device)
# for t in range(T):
#     spins = torch.tensor(np.random.choice([-1, 1], size=(n_realizations, 10), p=[0.5, 0.5]).astype(NP_DTYPE), device=device)
#     # spins = torch.ones(size=(n_realizations, 10), device=device)
#     log_avg_mag_cuda(avg_mag_1, t, spins, topology)
#     log_avg_mag_torch(avg_mag_2, t, spins.detach().cpu().numpy(), topology.detach().cpu().numpy())
#     log_avg_mag_cuda_2(avg_mag_3, t, spins, topology)
#     print(spins)
#     print(avg_mag_1[:, t])
#     print(avg_mag_2[:, t])
#     print(avg_mag_3[:, t])
#     print()



# def gen_J_pop(J_pop_size, topology):
#     gamma = 1.5
#     def gen_init_pop_f():
#         return create_J_C(J_C=None, same_J_C=False, J_C_sigma=gamma / np.sqrt(len(topology)), n_realizations=J_pop_size, topology=topology)
#     return gen_init_pop_f

# def avg_return_time_fitness(topology, n_realizations_per_J, T_w, sim_len, H, spins_p, n_spins,
#                             stable_t=STABLE_T, relaxation_t=None, n_std=0):  # TODO global constants?
#     if relaxation_t is None: relaxation_t = (stable_t[1] + T_w, sim_len)
#     device = torch.device(DEVICE)
#
#     F_idx = get_idx_arrays(topology)
#     F_idx = torch.as_tensor(F_idx, dtype=torch.long, device=device)
#     H = torch.as_tensor(H, dtype=TORCH_DTYPE, device=device)
#     topology = torch.as_tensor(topology, dtype=torch.long, device=device)
#     def fitness_f(population):
#         mean_return_time = np.zeros(len(population))
#         for J_C_i, J_C in enumerate(population):
#             avg_mag = torch.zeros(size=(n_realizations_per_J, sim_len), dtype=TORCH_DTYPE, device=device)
#             spins = create_spins(spins=None, same_init_spins=False, spins_p=spins_p,
#                              n_realizations=n_realizations_per_J, n_spins=n_spins)
#             spins = torch.as_tensor(spins, dtype=TORCH_DTYPE, device=device)
#             J_C = torch.as_tensor(J_C, dtype=TORCH_DTYPE, device=device)
#             _run_sim_torch(sim_len=sim_len, J_C=J_C, spins=spins, topology=topology, H=H,
#                            F_idx=F_idx, avg_mag=avg_mag, full_state=None)
#             return_times = get_return_times_jit(mag=avg_mag.detach().cpu().numpy(), stable_t=stable_t, relaxation_t=relaxation_t, n_std=n_std)
#             mean_return_time[J_C_i] = np.mean(return_times)
#         return mean_return_time
#     return fitness_f

# def avg_return_time_fitness(topology, n_realizations_per_J, T_w, sim_len, H, spins_p, n_spins,
#                             stable_t=STABLE_T, relaxation_t=None, n_std=0):  # TODO global constants?
#     if relaxation_t is None: relaxation_t = (stable_t[1] + T_w, sim_len)
#     device = torch.device(DEVICE)
#
#     F_idx = get_idx_arrays(topology)
#     F_idx = torch.as_tensor(F_idx, dtype=torch.long, device=device)
#     H = torch.as_tensor(H, dtype=TORCH_DTYPE, device=device)
#     topology = torch.as_tensor(topology, dtype=torch.long, device=device)
#     def fitness_f(population):
#         mean_return_time = np.zeros(len(population))
#         avg_mag = torch.zeros(size=(n_realizations_per_J*len(population), sim_len), dtype=TORCH_DTYPE, device=device)
#         spins = create_spins(spins=None, same_init_spins=False, spins_p=spins_p,
#                              n_realizations=n_realizations_per_J*len(population), n_spins=n_spins)
#         spins = torch.as_tensor(spins, dtype=TORCH_DTYPE, device=device)
#         J_C = torch.as_tensor(population, dtype=TORCH_DTYPE, device=device)
#         J_C = torch.repeat_interleave(J_C, n_realizations_per_J, dim=0)
#         _run_sim_torch(sim_len=sim_len, J_C=J_C, spins=spins, topology=topology, H=H,
#                        F_idx=F_idx, avg_mag=avg_mag, full_state=None)
#         return_times = get_return_times_jit(mag=avg_mag.detach().cpu().numpy(), stable_t=stable_t, relaxation_t=relaxation_t, n_std=n_std)
#         for J_C_i, J_C in enumerate(population):
#             mean_return_time[J_C_i] = np.mean(return_times[J_C_i*n_realizations_per_J:(J_C_i+1)*n_realizations_per_J])
#         return mean_return_time
#     return fitness_f


# def get_fitness_p(fitness, epsilon=0, increasing=True, log=False):
#     """
#     Get a probability vector from a fitness vector
#     :param fitness:
#     :param epsilon: If nonzero, this is the minimal probability of any entry
#     :param increasing: If true, increasing fitness equals increasing probability of being selected
#     :param log: log-transform before normalizing
#     :return:
#     """
#     fitness_p = fitness - np.min(fitness) if increasing else -fitness + np.max(fitness)
#     if log:
#         fitness_p += np.log2(fitness_p + 1)
#     fitness_p = fitness_p/np.sum(fitness_p)
#     if epsilon:
#         assert epsilon<1/len(fitness)
#         raise NotImplementedError()
#     return fitness_p

# def prop_selection_with_elitism(n_elite, epsilon=0):
#     # Lower fitness considered better
#     def selection_f(parents, parents_fitness, children, children_fitness):
#         population = np.concatenate([parents, children])
#         fitness = np.concatenate([parents_fitness, children_fitness])
#         fitness_p = get_fitness_p(fitness=fitness, epsilon=epsilon, increasing=False)
#         selected_idx = np.zeros(len(parents), dtype=np.int64)
#         selected_idx[:-n_elite] = np.random.choice(len(population), replace=True, p=fitness_p, size=len(parents)-n_elite)
#         selected_idx[-n_elite:] = np.argsort(fitness)[-n_elite:]
#         return population[selected_idx], fitness[selected_idx]
#     return selection_f

# def gaussian_mutation(n_mutated_entries, n_children_per_parent, mut_sigma):
#     def mutation_f(population):
#         n_loops = population.shape[-1]
#         epsilon = np.zeros(shape=(len(population)*n_children_per_parent, n_loops, n_loops))
#         for parent_i, parent in enumerate(population):
#             mut_rows = np.random.choice(n_loops, replace=False, size=n_mutated_entries)
#             mut_cols = np.random.choice(n_loops, replace=False, size=n_mutated_entries)
#             epsilon[parent_i, mut_rows, mut_cols] = mut_sigma * np.random.randn(n_mutated_entries)
#         return np.repeat(population, n_children_per_parent, axis=0) + epsilon
#     return mutation_f



# @jit(nopython=True, parallel=True)
# def update_C(J_C, C_spins, FC_spins, new_C_spins, H):
#     for i in prange(len(C_spins)):
#         new_C_spins[i] = np.sign(J_C[i]@C_spins[i] + FC_spins[i] + H)


# @util.time_func
# # @jit(nopython=True, parallel=True, nogil=True)
# @jit(nopython=True, parallel=True)
# def _run_sim_np(sim_len, J_C, spins, topology, H, C_idx, R_idx, FR_idx, FC_idx, avg_mag=None, full_state=None):
#     """
#     :param sim_len: Simulation time
#     :param J_C: A tensor (or ndarray) of shape (n_realizations, n_loops, n_loops). Overrides J_C_sigma and same_J_C
#     :param spins: A tensor (or ndarray) of shape (n_realizations, n_spins). Overrides J_C_sigma and same_J_C
#     :param topology: An array of loop sizes
#     :param H: An array of lengh sim_len with the size of h (the magnetic field) at each time point
#     :param C_idx: Indices of communicating spins
#     :param R_idx: Indices of non-communicating (rest) spins
#     :param FR_idx: Indices of spins feeding into communicating spins
#     :param FC_idx: Indices of spins feeding into non-communicating spins
#     :return: Results from the simulation run - A tensor of shape (n_realizations, sim_len) with the magnetization
#     for each realization at each time point (each loop is averaged  separately and these are then
#     also averaged - yielding a value in the range [-1, 1])
#     """
#     new_spins = np.zeros_like(spins)
#     n_realizations = len(spins)
#     for t in range(sim_len):
#         if avg_mag is not None: log_avg_mag(avg_mag, t, spins, C_idx=C_idx, topology=topology)
#         if full_state is not None: full_state[:, t, :] = spins
#         for r in prange(n_realizations):
#             new_spins[r][R_idx] = spins[r][FR_idx]
#             new_spins[r][C_idx] = np.sign(J_C[r] @ spins[r][C_idx] + spins[r][FC_idx] + H[t])
#         spins[...] = new_spins


# @util.time_func
# @profile
# def _run_sim_torch(sim_len, J_C, spins, topology, H, C_idx, R_idx, FR_idx, FC_idx,
#                    avg_mag=None, full_state=None):
#     new_spins = torch.empty_like(spins)
#     for t in range(sim_len):
#         if avg_mag is not None: log_avg_mag_torch(avg_mag, t, spins.detach().cpu().numpy(), C_idx=C_idx, topology=topology)
#         if full_state is not None: full_state[:, t, :] = spins
#         new_spins[:, R_idx] = spins[:, FR_idx]
#         new_spins[:, C_idx] = torch.sign(torch.matmul(J_C, spins[:, C_idx, None]).squeeze(-1) + spins[:, FC_idx] + H[t])
#         spins[...] = new_spins


# @cuda.jit
# def sum_spins_per_loop_cuda_2(loop_mag_sum, spins, topology, n_loops):
#     s = cuda.shared.array(shape=(TPB), dtype=float32)
#     row = cuda.blockIdx.x
#     tx = cuda.threadIdx.x
#     lower = n_loops
#     for loop_idx in range(n_loops):  # n_loops
#         upper = lower + topology[loop_idx] - 1
#         # val = float32(spins[row, loop_idx])  # C_spin
#         val = float32(0)  # C_spin
#         # s[i] = sum(loop[j] | j mod TPB = i)
#         for i in range(tx+lower, upper, TPB):  # Sum R_spins
#             val += spins[row, i]
#         s[tx] = val
#         lower = upper
#         # Sum all s to s[0] and assign s[0] to sum_arr[realization, loop_idx]
#         mid = BLOCK_MID
#         for i in range(TPB_LOG_2):
#             cuda.syncthreads()
#             if tx < mid:
#                 s[tx] += s[tx+mid]
#             mid >>= 1  # mid/=2
#         if tx == 0:
#             loop_mag_sum[row, loop_idx] = s[0]
#
# @profile
# def log_avg_mag_cuda_2(avg_mag, t, spins, topology):
#     n_loops, n_realizations, n_spins = len(topology), spins.shape[0], spins.shape[1]
#     loop_mag_sum = torch.zeros(size=(n_realizations, n_loops), device=torch.device("cuda:0"))  # TODO only create once (and then zero)
#     BPG = n_realizations # Blocks per grid
#     sum_spins_per_loop_cuda_2[BPG, TPB](loop_mag_sum, spins, topology, n_loops)
#     loop_mag_sum += spins[:, :n_loops]  # C_spins
#     loop_mag_sum /= topology  # divide by loop lengths to get mean
#     avg_mag[:, t] = torch.mean(loop_mag_sum, dim=1)

# @jit(nopython=True, parallel=True)
# def log_avg_mag_torch(avg_mag, t, spins, C_idx, topology):
#     n_realizations = len(spins)
#     for loop_c_i in prange(len(C_idx)):
#         loop_c = C_idx[loop_c_i]
#         for i in prange(n_realizations):
#             avg_mag[i, t] += np.mean(spins[i, loop_c:loop_c + topology[loop_c_i]])
#     avg_mag[:, t] /= len(topology)
# avg_mag[:, t] = np.mean([np.mean(spins[:, C_idx[loop_c_i]:C_idx[loop_c_i] + topology[loop_c_i]], axis=1) for loop_c_i in range(len(C_idx))], axis=0)
# lst = [torch.mean(spins[:, C_idx[loop_c_i]:C_idx[loop_c_i] + topology[loop_c_i]], dim=1) for loop_c_i in range(len(C_idx))]
# lst_torch = torch.as_tensor(lst, device=torch.device('cuda'))
# avg_mag[:, t] = torch.mean(lst_torch, dim=0)
# avg_mag[:, t] = torch.mean(torch.as_tensor([torch.mean(spins[:, C_idx[loop_c_i]:C_idx[loop_c_i] + topology[loop_c_i]], dim=1)
#                                             for loop_c_i in range(len(C_idx))], device=torch.device('cuda')), dim=0)



# def log_avg_mag(avg_mag, t, spins, C_idx, topology):
#     for loop_c_i, loop_c in enumerate(C_idx):
#         avg_mag[:, t] += np.mean(spins[:, loop_c:loop_c + topology[loop_c_i]], axis=1)
#     avg_mag[:, t] /= len(topology)


# @jit(nopython=True, parallel=True)
# def log_avg_mag(avg_mag, t, spins, C_idx, topology):
#     n_realizations = len(spins)
#     for loop_c_i in prange(len(C_idx)):
#         loop_c = C_idx[loop_c_i]
#         for i in prange(n_realizations):
#             avg_mag[i, t] += np.mean(spins[i, loop_c:loop_c + topology[loop_c_i]])
#     avg_mag[:, t] /= len(topology)


# @jit(nopython=True, parallel=True)
# def log_avg_mag_torch(avg_mag, t, spins, topology):
#     n_loops = len(topology)
#     cur_R_idx = n_loops
#     for loop_i in range(n_loops):
#         next_R_idx = cur_R_idx + topology[loop_i] - 1
#         for realization_i in prange(len(spins)):
#             avg_mag[realization_i, t] += (spins[realization_i, loop_i] + np.sum(spins[realization_i, cur_R_idx: next_R_idx])) / topology[loop_i]
#         cur_R_idx = next_R_idx
#     avg_mag[:, t] /= n_loops

#
# import numpy as np
# import util
# import torch
# from numba import jit, prange
# from exp_log import GloopNetExpLog
#
#
#
#
# def get_torch_idx_arrays(loop_sizes, device):
#     n_spins = np.sum(loop_sizes)
#     C_idx = np.zeros_like(loop_sizes)  # C spins (communicating)
#     C_idx[1:] = np.cumsum(loop_sizes[:-1])
#     mask = np.zeros(n_spins, dtype=bool)
#     mask[C_idx] = True
#     R_idx = np.arange(n_spins)[~mask]  # R spins (Rest - Non communicating)
#
#     # FC_idx = np.array([cs_ls - 1 for cs_ls in np.cumsum(loop_sizes)])  # spins feeding into C spins
#     # FR_idx = R_idx - 1  # spins feeding into R spins
#
#     # TODO yuck yuck yuck this is only for the sanity check with Shaked's code
#     FC_idx = np.array([C_idx[i] + 1 if loop_sizes[i] >1 else C_idx[i] for i in range(len(loop_sizes))])  # spins feeding into C spins
#     FR_idx = np.zeros_like(R_idx)
#     idx = np.zeros(len(loop_sizes)+1, dtype=np.int32)  # C spins (communicating)
#     idx[1:] = np.cumsum(loop_sizes)
#     x = np.arange(n_spins)
#     cur_i=0
#     for i, loop_size in enumerate(loop_sizes):
#         if loop_size>1:
#             cur = x[idx[i]+1:idx[i+1]]+1
#             cur[-1] = C_idx[i]
#             FR_idx[cur_i:cur_i+len(cur)] = cur
#             cur_i+=len(cur)
#
#
#
#
#     R_idx = torch.tensor(R_idx, dtype=torch.long, device=device)
#     FR_idx = torch.tensor(FR_idx, dtype=torch.long, device=device)
#     C_idx = torch.tensor(C_idx, dtype=torch.long, device=device)
#     FC_idx = torch.tensor(FC_idx, dtype=torch.long, device=device)
#
#     return C_idx, R_idx, FC_idx, FR_idx
#
#
# def log_results(results, t, spins, C_idx, loop_sizes):
#     for loop_c_i, loop_c in enumerate(C_idx):
#         results[:, t] += torch.mean(spins[:, loop_c:loop_c + loop_sizes[loop_c_i]], dim=1)
#     results[:, t] /= len(loop_sizes)
#
#
# def create_J_C(J_C, same_J_C, J_C_sigma, n_realizations, n_loops, device):
#     if J_C is not None:  # Specific value passed. Not randomized
#         J_C = torch.tensor(J_C, dtype=torch.float32, device=device)
#     elif same_J_C:  # Randomize one J_C across realizations
#         J_C = J_C_sigma * torch.randn(n_loops, n_loops, dtype=torch.float32, device=device)
#         J_C[np.diag_indices(J_C.shape[0])] = 0
#         # J_C = J_C.expand(n_realizations, *J_C.shape)  # TODO temporary hack (should be broadcasted)
#     else:
#         # TODO "CUDA out of memory" occurs here if too many realizations (consecutive loops?)
#         J_C = J_C_sigma * torch.randn(n_realizations, n_loops, n_loops, dtype=torch.float32, device=device)
#     return J_C  # TODO zero diagonal!
#
#
# def create_spins(spins, same_init_spins, spins_p, n_realizations, n_spins, device):
#     if spins is not None:  # Specific value passed. Not randomized
#         spins = torch.tensor(spins, device=device, dtype=torch.float32)
#         if len(spins.shape) == 1:
#             spins = spins.expand(n_realizations, *spins.shape)
#     elif same_init_spins:  # Randomize one initial conditions across realizations
#         spins = np.random.choice([-1, 1], size=n_spins, p=[1 - spins_p, spins_p])
#         spins = torch.tensor(spins, device=device, dtype=torch.float32)
#         spins = spins.expand(n_realizations, *spins.shape)
#     else:
#         if spins_p is None:  # New p for each realization
#             raise NotImplementedError("Different initial conditions with different p not implemented yet")  # TODO
#         else:
#             spins = np.random.choice([-1, 1], size=(n_realizations, n_spins), p=[1 - spins_p, spins_p])
#             spins = torch.tensor(spins, device=device, dtype=torch.float32)
#     return spins
#
# @util.time_func
# def _run_sim_torch(n_realizations, sim_len, J_C, spins, loop_sizes, H, C_idx, R_idx, FR_idx, FC_idx, device,
#                    log_full_state=False):
#     """
#     :param n_realizations: Number of realizations
#     :param sim_len: Simulation time
#     :param J_C: A tensor (or ndarray) of shape (n_realizations, n_loops, n_loops). Overrides J_C_sigma and same_J_C
#     :param spins: A tensor (or ndarray) of shape (n_realizations, n_spins). Overrides J_C_sigma and same_J_C
#     :param loop_sizes: An array of loop sizes
#     :param H: An array of lengh sim_len with the size of h (the magnetic field) at each time point
#     :param C_idx: Indices of communicating spins
#     :param R_idx: Indices of non-communicating (rest) spins
#     :param FR_idx: Indices of spins feeding into communicating spins
#     :param FC_idx: Indices of spins feeding into non-communicating spins
#     :param device: String (cuda:0 or cpu) with the device on which to run the simulation
#     :param log_full_state: If true, the full state (all spins) at each time point
#     :return: Results from the simulation run - A tensor of shape (n_realizations, sim_len) with the magnetization
#     for each realization at each time point (each loop is averaged  separately and these are then
#     also averaged - yielding a value in the range [-1, 1])
#     """
#     if log_full_state: state = torch.zeros(size=(n_realizations, sim_len, np.sum(loop_sizes)), device=device, dtype=torch.int32)
#     results = torch.zeros(size=(n_realizations, sim_len), device=device, dtype=torch.float32)
#     new_spins = torch.zeros_like(spins)
#     for t in range(sim_len):
#         log_results(results, t, spins, C_idx=C_idx, loop_sizes=loop_sizes)
#         if log_full_state: state[:, t, :] = spins
#         # new_spins = torch.zeros_like(spins)
#         new_spins[:, R_idx] = spins[:, FR_idx]
#         # new_spins[:, C_idx] = torch.sign(torch.matmul(J_C, spins[:, C_idx, None])[..., 0] + spins[:, FC_idx] + H[t])
#         new_spins[:, C_idx] = torch.sign(torch.matmul(J_C, spins[:, C_idx].T).T + spins[:, FC_idx] + H[t])
#         spins = torch.clone(new_spins)
#
#     if log_full_state: return results, state
#     return results
#
#
# def run_sim_torch(exp_log: GloopNetExpLog):
#     """
#     Setup relevant tensors, run simulations and save results
#     :param exp_log: An GloopNetExpLog object with all relevant parameters for the simulation/s
#     """
#     # Generate required tensors
#     device = torch.device(exp_log.device)
#     n_loops, n_spins = len(exp_log.loop_sizes), np.sum(exp_log.loop_sizes)
#     C_idx, R_idx, FC_idx, FR_idx = get_torch_idx_arrays(exp_log.loop_sizes, device)
#     J_C = create_J_C(J_C=exp_log.J_C, same_J_C=exp_log.same_J_C, J_C_sigma=exp_log.J_C_sigma,
#                      n_realizations=exp_log.n_realizations, n_loops=n_loops, device=device)
#     spins = create_spins(spins=exp_log.spins, same_init_spins=exp_log.same_init_spins, spins_p=exp_log.spins_p,
#                          n_realizations=exp_log.n_realizations, n_spins=n_spins, device=device)
#     # Annnnd... run
#
#     for (sim_len, H) in zip(exp_log.sim_len, exp_log.H):
#         results = _run_sim_torch(n_realizations=exp_log.n_realizations, sim_len=sim_len, J_C=J_C, spins=spins,
#                                  loop_sizes=exp_log.loop_sizes, H=H,
#                                  C_idx=C_idx, R_idx=R_idx, FR_idx=FR_idx, FC_idx=FC_idx, device=device, log_full_state=False)  # TODO fix this ugly (log_full_state)
#         # np.save(f"{exp_log.dir}\\state_sim_len={sim_len}", state.cpu().numpy())
#         np.save(f"{exp_log.dir}\\mag_results_sim_len={sim_len}", results.cpu().numpy())
#
#
# @jit(nopython=True, parallel=True)
# def get_init_spins_diff_p(n_realizations, n_spins):
#     # TODO different p for initial condition. Not tested yet
#     ps = np.random.uniform(0, 1, n_realizations)
#     spins = np.empty(n_realizations, n_spins)
#     for i in prange(n_realizations):
#         spins[i] = np.random.choice([-1, 1], n_spins, p=[1 - ps[i], ps[i]])
#
#
# @jit(nopython=True, parallel=True)
# def get_return_times_jit(mag, stable_t, relaxation_t, n_std):
#     """
#     :param mag: Tensor of size (n_realizations, sim_len) with avg. of loop magnetizations (average per loop averaged over loops)
#     :param stable_t: Tuple with start/end times of period where system is considered in ground state.
#     :param relaxation_t: Tuple with start/end times in which to search for return to ground state (will report number of
#             time steps until magnetization returns to mean+n_std*sigma, computed over stable_t)
#     :param n_std: Return to ground state is considered when magnetization crosses the threshold mean+n_std*sigma
#     :return: Return times to ground state for each realization
#     """
#     n_realizations, sim_len = mag.shape
#     return_times = np.zeros(n_realizations, dtype=np.int64) # TODO default should not be 0
#
#     def get_return_time(i, thresh):
#         for t in prange(relaxation_t[0], relaxation_t[1]):
#             if mag[i, t] < thresh:
#                 return_times[i] = t - relaxation_t[0]  # TODO batch
#                 return
#
#     for i in prange(n_realizations):
#         mean = np.mean(mag[i, stable_t[0]:stable_t[1]])
#         std = np.std(mag[i, stable_t[0]:stable_t[1]])
#         thresh = mean + n_std * std
#         get_return_time(i, thresh)
#
#     return return_times
#
#
# def main():
#     for i in range(1, 2):
#         loop_sizes = np.load(f"JInfo_secondTry\\loop_sizes_{i}.npy")
#         # J_C = np.load("J_ij_reduced.npy")
#         # J_C[np.diag_indices(J_C.shape[0])] = 0
#         # J_C = J_C[None, :, :]
#         # init_spins = ((np.load("spins_history.npy")[:, 0])*2 -1)[None, :]
#
#         # T_ws = [20, 40, 160, 640, 900, 1280, 2100, 3000]
#         # T_ws = [3000]
#         # sim_lens = [6000 + T_w for T_w in T_ws]
#         # Hs = [np.zeros(sim_len) for sim_len in sim_lens]
#         # for H, T_w in zip(Hs, T_ws): H[2000:2000+T_w] = 0.2
#
#         sim_lens = [100]
#         Hs = [np.zeros(sim_len) for sim_len in sim_lens]
#
#         gamma = 1.5
#         J_sigma = gamma/np.sqrt(len(loop_sizes))
#         exp_log = GloopNetExpLog \
#                 (
#                 # exp_name=f"Shaked_2^18_spins_same_J_ls_{i}",
#                 exp_name=f"tst_{i}",
#                 n_realizations=900,  # Scalar
#                 sim_len=sim_lens,  # Scalar or list for multiple runs
#                 H=Hs,  # 1D array (sim_len), 2D array (sim_len, n_loops) or list of arrays for multiple runs  (TODO replace with T_w?)
#                 loop_sizes=loop_sizes,  # 1D array defining network topology
#                 J_C=None,  # 2D array (n_loops, n_loops) of connecting spin interaction weight or None for randomizing
#                 same_J_C=True,  # (If J_C is None) Randomize J_C once or for each realization
#                 J_C_sigma=J_sigma,  # Randomized J_C connections are drawn from N(0, sigma*I)
#                 spins=None,  # 1D array (n_spins) of initial values for spins or None for randomizing
#                 same_init_spins=False,  # (If spins is None) Randomize initial conditions once or for each realization
#                 spins_p=0.5,  # Randomized J_C connections are drawn from Bin(p) (if None, and same_init_spins=False,
#                 # a new p~uni(0, 1) is drawn for each realization)
#                 device="cuda:0"
#             )
#         run_sim_torch(exp_log=exp_log)
#         # Analyze results (get return times)
#         # exp_log = GloopNetExpLog.load("2021_06_09_00_09_04_test")
#         # for sim_len, T_w in zip(sim_lens, T_ws):
#         #     mag_results = np.load(f"{exp_log.dir}\\mag_results_sim_len={sim_len}.npy")
#         #     return_times = get_return_times_jit(mag=mag_results, stable_t=(1000, 2000), relaxation_t=(2000+T_w, sim_len), n_std=0)
#         #     np.save(f"{exp_log.dir}\\return_times_results_sim_len={sim_len}", return_times)
#
# if __name__ == '__main__':
#     main()
