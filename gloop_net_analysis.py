import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from gloop_net_sim import run_RCCN, create_spins, get_feeding_idx
import util
import gloop_net_evolution as gne
import time
from scipy.stats import pearsonr
from collections import namedtuple
import pickle
from scipy.stats import gaussian_kde

DEBUG = False


def create_gif(plot_frame_i, n_frames, gif_path, lag=1):
    if not os.path.isdir("gif_temp"): os.makedirs("gif_temp")
    filenames = []
    for i in range(n_frames):
        if i%lag!=0: continue
        plot_frame_i(i)
        filename = f"gif_temp\\{i}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()
    # build gif
    with imageio.get_writer(f"{gif_path}.gif", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


def generate_lag_times_for_best_solution_i(exp_name, evo_i, log_freq=1):
    if evo_i%log_freq != 0: return
    exp_dir = f"experiments\\EvoGloop\\{exp_name}"
    best_solutions = np.load(f"{exp_dir}\\best_solutions.npy")
    topology = np.load(f"{exp_dir}\\topology.npy")
    mag_loops = np.load(f"{exp_dir}\\mag_loops.npy")
    if not os.path.isdir(f"{exp_dir}\\lag_times"): os.makedirs(f"{exp_dir}\\lag_times")
    n_spins = 2**14 if not DEBUG else 2**8
    spins_p = 0.5
    for T_w in util.T_WS:
        sim_len = util.STABLE_T[1] + T_w + util.RELAXATION_TIME
        H = np.zeros(shape=(sim_len, len(topology)))
        H[util.STABLE_T[1]:util.STABLE_T[1]+T_w, mag_loops] = gne.mag_field_strength(n_spins)
        fitness_f = gne.LagTimeFitness(topology, n_realizations_per_J=900, T_w=T_w, sim_len=sim_len,
                                       H=H, spins_p=spins_p, n_spins=n_spins)
        lag_times = fitness_f(population=best_solutions[evo_i//log_freq][None, ...])
        np.save(f"{exp_dir}\\lag_times\\best_{evo_i}_partial_mag_T_w={T_w}.npy", lag_times)
        H.fill(0)
        H[util.STABLE_T[1]:util.STABLE_T[1]+T_w, :] = gne.mag_field_strength(n_spins)
        fitness_f = gne.LagTimeFitness(topology, n_realizations_per_J=900, T_w=T_w, sim_len=sim_len,
                                       H=H, spins_p=spins_p, n_spins=n_spins)
        lag_times = fitness_f(population=best_solutions[evo_i//log_freq][None, ...])
        np.save(f"{exp_dir}\\lag_times\\best_{evo_i}_full_mag_T_w={T_w}.npy", lag_times)


def gen_kill_curve_gif(exp_name, lag, xscale="linear"):
    exp_dir = f"experiments\\EvoGloop\\{exp_name}"
    fitness_traj = np.load(f"{exp_dir}\\fitness_traj.npy")
    n_iterations = len(fitness_traj) * lag
    print(f"{exp_name}")
    print(f"Best fitness went from {np.min(fitness_traj[0])} to {np.min(fitness_traj[-1])}")
    print(f"Mean fitness went from {np.mean(fitness_traj[0])} to {np.mean(fitness_traj[-1])}")
    def plot_frame_i(i):
        # Plot Kill curve
        print(f"plotting frame {i}")
        with plt.style.context("dark_background"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"iteration {i}", fontsize=25, c="c")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        for T_w_i, T_w in zip(np.linspace(0.3, 1, len(util.T_WS)), util.T_WS):
            full_mag_lag_times = np.load(f"{exp_dir}\\lag_times\\best_{i}_full_mag_T_w={T_w}.npy")
            ax1.set(title="Full magnetic field kill curve", xscale=xscale, yscale="log", xlabel="Return-time (to ground state)", ylabel="1-CDF", ylim=(0.01, 1.1), xlim=(1, 800))
            ax1.plot(np.sort(full_mag_lag_times), 1-np.linspace(0, 1, len(full_mag_lag_times), endpoint=False), c=util.cmap(T_w_i), label=f"T_w={T_w}")
            partial_mag_lag_times = np.load(f"{exp_dir}\\lag_times\\best_{i}_partial_mag_T_w={T_w}.npy")
            ax2.set(title="Partial magnetic field kill curve", xscale=xscale, yscale="log", xlabel="Return-time (to ground state)", ylim=(0.01, 1.1), xlim=(1, 800))
            ax2.plot(np.sort(partial_mag_lag_times), 1-np.linspace(0, 1, len(partial_mag_lag_times), endpoint=False), c=util.cmap(T_w_i), label=f"T_w={T_w}")
            plt.legend()
    create_gif(plot_frame_i=plot_frame_i, n_frames=n_iterations, gif_path=f"figure_dump\\gif_{exp_name}", lag=lag)


def plot_kill_curves(lag_times, xscale="linear", yscale="log", title="", save_dir=""):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    if title: fig.suptitle(title, fontsize=25, c="c")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for T_w_i, T_w in zip(np.linspace(0.3, 1, len(lag_times)), lag_times):
        ax.set(title="kill curve", xscale=xscale, yscale=yscale, xlabel="Return-time (to ground state)", ylabel="1-CDF", ylim=(0.01, 1.1), xlim=(1, 800))
        ax.plot(np.sort(lag_times[T_w]), 1-np.linspace(0, 1, len(lag_times[T_w]), endpoint=False), c=util.cmap(T_w_i), label=f"T_w={T_w}")
    plt.legend()
    util.save_fig(save_dir, figure_dump=False, facecolor="k") if save_dir else plt.show()


def plot_lag_distributions(lag_times, xscale="linear", x_max=None, y_max=None, title="", save_dir=""):
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    if title: fig.suptitle(title, fontsize=25, c="c")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    bw=None
    for T_w_i, T_w in zip(np.linspace(0.3, 1, len(lag_times)), lag_times):
        ax.set(title="Lag distribution", xscale=xscale, yscale="linear", xlabel="Return-time (to ground state)", ylabel="PDF", xlim=(1, x_max) if x_max else None, ylim=(-1, y_max) if y_max else None)
        lag_times_T_w = lag_times[T_w]
        if x_max: lag_times_T_w = lag_times_T_w[lag_times_T_w<=x_max]
        t_range = np.linspace(1, x_max if x_max is not None else 800)
        n, *_ = ax.hist(lag_times_T_w, color=util.cmap(T_w_i), label=f"T_w={T_w}", alpha=0.8, density=False, bins=x_max if x_max is not None else 40)
        # kde = gaussian_kde(lag_times_T_w, bw_method=bw)
        # kde_est = kde(t_range)
        # scalar = max(n) / max(kde_est)
        # ax.plot(t_range, kde_est*scalar, lw=2, c=util.cmap(T_w_i))

    plt.legend()
    util.save_fig(save_dir, figure_dump=False, facecolor="k") if save_dir else plt.show()


def get_lag_times(topology, J, mag_loops, n_realizations=900, spins_p=0.5, T_ws=util.T_WS):
    n_spins = np.sum(topology)
    lag_times = {}
    for T_w in T_ws:
        sim_len = util.STABLE_T[1] + T_w + util.RELAXATION_TIME
        H = np.zeros(shape=(sim_len, len(topology)))
        H[util.STABLE_T[1]:util.STABLE_T[1]+T_w, mag_loops] = gne.mag_field_strength(n_spins)
        fitness_f = gne.LagTime_F(topology, n_realizations_per_J=n_realizations, T_w=T_w, sim_len=sim_len,
                                       H=H, spins_p=spins_p, n_spins=n_spins)
        lag_times[T_w] = fitness_f(population=J[None, ...])
        print(f"Computed return times for T_w={T_w}")
    return lag_times


def divide_by_size(topology, n_bins):
    """
    Return n_bins index arrays that divide a's unique elements into n_bins equal bins
    """
    borders = [np.percentile(topology, i*100/n_bins) for i in range(n_bins+1)]
    borders[0] = 0
    idx_arrays = [np.argwhere((borders[i]<topology)&(topology<=borders[i+1])).flatten() for i in range(len(borders)-1)]
    return idx_arrays


class JTrajAnalysis:
    def __init__(self):
        self.all_traj_tests = \
            {
                "Mean": lambda traj: np.mean(traj, axis=(1, 2)),
                "Std": lambda traj: np.std(traj, axis=(1, 2)),
                "Symmetry": self.symmetry,
                "2_cycles": self.cycles_2,
                "3_cycles": self.cycles_3,
                "4_cycles": self.cycles_4,
            }


    @staticmethod
    def _is_symmetric(traj):
        return traj.shape[-2] == traj.shape[-1]

    def symmetry(self, traj):
        if not JTrajAnalysis._is_symmetric(traj): return None # Not a square matrix
        sym = lambda A: pearsonr(A.flatten(), A.T.flatten())[0]
        return np.array([sym(A) for A in traj])

    def cycles_2(self, traj):
        if not JTrajAnalysis._is_symmetric(traj): return None # Not a square matrix
        n_cycles_2 = lambda A: np.sum(np.diag(A@A))  # Assuming A represents a connectivity matrix
        threshold_traj = np.zeros_like(traj)
        threshold_traj[-traj > self.connectivity_threshold] = 1
        return np.array([n_cycles_2(A) for A in threshold_traj])

    def cycles_3(self, traj):
        if not JTrajAnalysis._is_symmetric(traj): return None # Not a square matrix
        n_cycles_3 = lambda A: np.sum(np.diag(A@A@A))  # Assuming A represents a connectivity matrix
        threshold_traj = np.zeros_like(traj)
        threshold_traj[-traj > self.connectivity_threshold] = 1
        return np.array([n_cycles_3(A) for A in threshold_traj])

    def cycles_4(self, traj):
        if not JTrajAnalysis._is_symmetric(traj): return None # Not a square matrix
        n_cycles_3 = lambda A: np.sum(np.diag(A@A@A@A)) - np.sum(np.diag(A@A)**2)  # Assuming A represents a connectivity matrix
        threshold_traj = np.zeros_like(traj)
        threshold_traj[-traj > self.connectivity_threshold] = 1
        return np.array([n_cycles_3(A) for A in threshold_traj])

    def corr_to_loop_size(self, traj, topology):
        raise NotImplemented


    def analyze_J_traj(self, tests_to_perform, traj, topology, connectivity_threshold=None, idx_groups=None):
        n_recorded_iterations, n_loops, _ = traj.shape
        if tests_to_perform is None: tests_to_perform = set(self.all_traj_tests.keys())
        self.connectivity_threshold = connectivity_threshold if connectivity_threshold is not None else np.percentile(np.abs(traj), 90)
        assert tests_to_perform.issubset(self.all_traj_tests)
        results = {test_name: None for test_name in tests_to_perform}
        for test_name in tests_to_perform:
            if idx_groups is None:  # Analyze full matrix
                results[test_name] = self.all_traj_tests[test_name](traj)
            else:  # Analyze sub_matrices
                for idx_group_name in idx_groups:
                    idx_group_i, idx_group_j = idx_groups[idx_group_name]
                    sub_traj = traj[np.ix_(np.arange(n_recorded_iterations), idx_group_i, idx_group_j)]
                    results[f"{test_name}_{idx_group_name}"] = self.all_traj_tests[test_name](sub_traj)
        return results




def best_sol_analysis(exp_name, upto_mag_idx=None):
    exp_dir = f"experiments\\EvoGloop\\{exp_name}"
    best_solutions = np.load(f"{exp_dir}\\best_solutions.npy")
    topology = np.load("topologies_and_Js\\topology_1.npy")
    small_mag_idx, med_mag_idx, large_mag_idx = divide_by_size(topology[:upto_mag_idx], n_bins=3)
    small_non_mag_idx, med_non_mag_idx, large_non_mag_idx = [_ + upto_mag_idx for _ in divide_by_size(topology[upto_mag_idx:], n_bins=3)]

    n_recorded_iterations, n_loops, _ = best_solutions.shape
    mag_idx = np.arange(0, upto_mag_idx)
    non_mag_idx = np.arange(upto_mag_idx, n_loops)

    corr = lambda A: pearsonr(A.flatten(), A.T.flatten())[0]
    dist_from_T = lambda A: np.mean(np.abs(A.T - A))
    Result = namedtuple('Result', ['arr', 'get_result'])
    results_d = \
        {
            "ingoing_mean": Result(arr=np.zeros([n_recorded_iterations, n_loops]), get_result=lambda J, *args :np.mean(J, axis=1)),
            "outgoing_mean": Result(arr=np.zeros([n_recorded_iterations, n_loops]), get_result=lambda J, *args :np.mean(J, axis=0)),
            "ingoing_std": Result(arr=np.zeros([n_recorded_iterations, n_loops]), get_result=lambda J, *args :np.std(J, axis=1)),
            "outgoing_std": Result(arr=np.zeros([n_recorded_iterations, n_loops]), get_result=lambda J, *args :np.std(J, axis=0)),
            "inout_corr": Result(arr=np.zeros([n_recorded_iterations, n_loops]), get_result=lambda J, *args: np.array([pearsonr(J[i], J[:, i])[0] for i in range(n_loops)])),
            "minor_means": Result(arr=np.zeros([n_recorded_iterations, 5]), get_result=lambda J, J_ul, J_ur, J_ll, J_lr: [np.mean(A) for A in [J, J_ul, J_ur, J_ll, J_lr]]),
            "minor_stds": Result(arr=np.zeros([n_recorded_iterations, 5]), get_result=lambda J, J_ul, J_ur, J_ll, J_lr: [np.std(A) for A in [J, J_ul, J_ur, J_ll, J_lr]]),
            "minor_corr": Result(arr=np.zeros([n_recorded_iterations, 3]), get_result=lambda J, J_ul, J_ur, J_ll, J_lr: [corr(J), corr(J_ul), corr(J_lr)]),
            "minor_dist_from_T": Result(arr=np.zeros([n_recorded_iterations, 3]), get_result=lambda J, J_ul, J_ur, J_ll, J_lr: [dist_from_T(J), dist_from_T(J_ul), dist_from_T(J_lr)]),
        }

    for evo_i in range(0, n_recorded_iterations):
        J = best_solutions[evo_i]
        J_ul, J_ur, J_ll, J_lr = J[np.ix_(mag_idx, mag_idx)], J[np.ix_(mag_idx, non_mag_idx)], J[np.ix_(non_mag_idx, mag_idx)], J[np.ix_(non_mag_idx, non_mag_idx)]
        for result in results_d:
            results_d[result].arr[evo_i] = results_d[result].get_result(J, J_ul, J_ur, J_ll, J_lr)

    mag_stats, non_mag_stats = {}, {}
    for results_name in ["ingoing_mean", "outgoing_mean", "ingoing_std", "outgoing_std", "inout_corr"]:
        mag_stats[f"{results_name}_mean"] = [np.mean(results_d[results_name].arr[:, idx], axis=1) for idx in [small_mag_idx, med_mag_idx, large_mag_idx]]
        mag_stats[f"{results_name}_std"] = [np.std(results_d[results_name].arr[:, idx], axis=1) for idx in [small_mag_idx, med_mag_idx, large_mag_idx]]
        non_mag_stats[f"{results_name}_mean"] = [np.mean(results_d[results_name].arr[:, idx], axis=1) for idx in [small_non_mag_idx, med_non_mag_idx, large_non_mag_idx]]
        non_mag_stats[f"{results_name}_std"] = [np.std(results_d[results_name].arr[:, idx], axis=1) for idx in [small_non_mag_idx, med_non_mag_idx, large_non_mag_idx]]


    pickle.dump(mag_stats, open(f"analysis\\{exp_name}_J_mag_stats", 'wb'))
    pickle.dump(non_mag_stats, open(f"analysis\\{exp_name}_J_non_mag_stats", 'wb'))

    # np.save(f"{exp_dir}\\something_{evo_i}.npy", something)
    for results_name in results_d:
        np.save(f"analysis\\{exp_name}_J_{results_name}.npy", results_d[results_name].arr)


def best_sol_plot(exp_name, log_freq=1):
    mag_results = pickle.load(open(f"shani's_sandbox\\{exp_name}_J_mag_stats", 'rb'))
    non_mag_results = pickle.load(open(f"shani's_sandbox\\{exp_name}_J_non_mag_stats", 'rb'))

    # minor_means = np.load(f"shani's_sandbox\\{exp_name}_J_minor_means.npy")
    # minor_stds = np.load(f"shani's_sandbox\\{exp_name}_J_minor_stds.npy")
    # minor_corr = np.load(f"shani's_sandbox\\{exp_name}_J_minor_corr.npy")
    # minor_dist_from_T = np.load(f"shani's_sandbox\\{exp_name}_J_minor_dist_from_T.npy")

    with plt.style.context("dark_background"):
        #fig, (ax_means, ax_stds, ax_corr, ax_dist_from_T) = plt.subplots(4, 1, figsize=(15, 10))
        fig, [[ax_ingoing_mean, ax_ingoing_std, _], [ax_outgoing_mean, ax_outgoing_std, ax_inout_corr]] =\
            plt.subplots(2, 3, figsize=(15, 10))

    # ax.set(xlabel="Time after T_w", ylabel="Magnetization")
    #for minor_i, minor_name in enumerate(["J", "ul", "lr", "ur", "ll"]):
    x_axis = np.arange(0, len(mag_results["ingoing_mean_mean"][0])*log_freq, log_freq)
    for idx_i, idx_name in enumerate(["small", "med", "large"]):
        ax_ingoing_mean.errorbar(x=x_axis, y=mag_results["ingoing_mean_mean"][idx_i], yerr=mag_results["ingoing_mean_std"][idx_i], label=f"{idx_name}_mean")
        ax_outgoing_mean.errorbar(x=x_axis, y=mag_results["outgoing_mean_mean"][idx_i], yerr=mag_results["outgoing_mean_std"][idx_i], label=f"{idx_name}_mean")
        ax_ingoing_std.errorbar(x=x_axis, y=mag_results["ingoing_std_mean"][idx_i], yerr=mag_results["ingoing_std_std"][idx_i], label=f"{idx_name}_std")
        ax_outgoing_std.errorbar(x=x_axis, y=mag_results["outgoing_std_mean"][idx_i], yerr=mag_results["outgoing_std_std"][idx_i], label=f"{idx_name}_std")
        ax_inout_corr.errorbar(x=x_axis, y=mag_results["inout_corr_mean"][idx_i], yerr=mag_results["inout_corr_std"][idx_i], label=f"{idx_name}_corr")

    #
    #     ax_means.plot(np.arange(0, len(minor_means)*log_freq, log_freq), minor_means[:, minor_i], label=f"{minor_name}_mean")
    #     ax_stds.plot(np.arange(0, len(minor_stds)*log_freq, log_freq), minor_stds[:, minor_i], label=f"{minor_name}_std")
    #     if minor_name not in {"ur", "ll"}:
    #         ax_corr.plot(np.arange(0, len(minor_corr)*log_freq, log_freq), minor_corr[:, minor_i], label=f"{minor_name}_corr")
    #         ax_dist_from_T.plot(np.arange(0, len(minor_dist_from_T)*log_freq, log_freq), minor_dist_from_T[:, minor_i], label=f"{minor_name}_dist_from_T")
    for ax in [ax_ingoing_mean, ax_ingoing_std, ax_outgoing_mean, ax_outgoing_std, ax_inout_corr]:
        ax.legend()
    # plt.show()
    plt.savefig(f"shani's_sandbox\\{exp_name}_J_stats")

def mag_field_strength(n_spins):
    return 0.8/(np.sqrt(n_spins/2**14))


def plot_trace(trace, T_w, lag_start_time):
    # trace_avg = np.mean(trace, dim=0)
    # trace_std = np.std(trace, dim=0)


    # ax.plot(np.arange(trace_avg.shape[0]), trace_avg)

    trace_len, n_traces = 5000, 2
    x = np.arange(trace_len)
    # y = trace_avg[:trace_len, :n_traces]
    # y_std = trace_std[:trace_len, :n_traces]
    for i in range(n_traces):
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.plot(x, np.mean(trace[:, :trace_len, i], axis=0))
        ax.plot(x, trace[:, :trace_len, i].T, alpha=0.1)
        # ax.fill_between(x, y[:, i] - y_std[:, i], y[:, i] + y_std[:, i], alpha=0.2)

        ax.axvline(x=lag_start_time, label="Lag start time")
        fig.suptitle(f"T_w={T_w}", fontsize=25, c="c")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.legend()
        plt.show()


def run_J(topology, J, n_realizations_per_J, T_ws, record_mag=False):
    spins_p = 0.5
    device = torch.device("cuda:0")
    n_spins = np.sum(topology)
    # F_idx = get_feeding_idx(topology)
    C_idx, _, FC_idx, _ = get_feeding_idx(topology, contiguous_C=False, return_only_F=False)
    C_idx, FC_idx = torch.as_tensor(C_idx, dtype=torch.long, device=device), torch.as_tensor(FC_idx, dtype=torch.long, device=device)
    topology, J = torch.as_tensor(topology, dtype=torch.long, device=device), torch.as_tensor(J, device=device)

    sim_lens = [util.STABLE_T[1] + T_w + util.RELAXATION_TIME for T_w in T_ws]
    Hs = [torch.as_tensor(np.zeros(shape=(sim_len, len(topology))), device=device) for sim_len in sim_lens]
    for H, T_w in zip(Hs, T_ws): H[util.STABLE_T[1]:util.STABLE_T[1] + T_w, :] = mag_field_strength(n_spins)

    J_C = torch.as_tensor(J, dtype=torch.float32, device=device)
    mag_records, lag_times = {}, {}
    for H, T_w in zip(Hs, T_ws):
        spins = create_spins(spins=None, same_init_spins=False, spins_p=spins_p,
                             n_realizations=n_realizations_per_J, n_spins=n_spins)
        spins = torch.as_tensor(spins, dtype=torch.float32, device=device)
        if record_mag:
            loop_mag_rec = torch.zeros(size=(n_realizations_per_J, len(H), len(topology)), dtype=torch.float32, device=device)
            lag_times[T_w] = run_RCCN(J_C=J_C, spins=spins, topology=topology, H=H, C_idx=C_idx, FC_idx=FC_idx, lag_start_time=util.STABLE_T[1] + T_w, loop_mag_rec=loop_mag_rec).detach().cpu().numpy()
            mag_records[T_w] = loop_mag_rec.detach().cpu().numpy()
        else:
            lag_times[T_w] = run_RCCN(J_C=J_C, spins=spins, topology=topology, H=H, C_idx=C_idx, FC_idx=FC_idx, lag_start_time=util.STABLE_T[1] + T_w, loop_mag_rec=None).detach().cpu().numpy()
    return (lag_times, mag_records) if record_mag else lag_times


    # plot_trace(trace=trace, T_w=T_w, lag_start_time=util.STABLE_T[1] + T_w)
    # with plt.style.context("dark_background"):
    #     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # fig.suptitle("title", fontsize=25, c="c")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # xscale = "linear"
    # ax.set(title="kill curve", xscale=xscale, yscale="log", xlabel="Return-time (to ground state)", ylabel="1-CDF", ylim=(0.01, 1.1), xlim=(1, 800))
    # for T_w, T_w_c, lag_time in zip(T_ws, np.linspace(0.3, 1, len(T_ws)), lag_times):
    #     ax.plot(np.sort(lag_time), 1-np.linspace(0, 1, len(lag_time), endpoint=False), c=util.cmap(T_w_c), label=f"T_w={T_w}")
    #     # ax.plot(np.arange(len(lag_times_hist)), 1-np.cumsum(lag_times_hist)/np.sum(lag_times_hist), c=util.cmap(T_w_c), label=f"T_w={T_w}")
    # plt.legend()
    # plt.show()
    # util.save_fig(save_dir, figure_dump=False, facecolor="k") if save_dir else plt.show()



#
# if __name__ == '__main__':
#     mpl.use("TkAgg")
#     all_results = {}
#
#     for i, exp_name in enumerate(os.listdir(f"experiments\\EvoGloop")):
#
#
#         best_solutions = np.load(f"experiments\\EvoGloop\\{exp_name}\\best_solutions.npy")
#         topology = np.load("topologies_and_Js\\topology_1.npy")
#         # small_idx, med_idx, large_idx = divide_by_size(topology, n_bins=3)
#         small_idx, large_idx = np.where(topology<150)[0], np.where(topology>=150)[0]
#         all_idx = np.arange(len(topology))
#         idx_groups = \
#             {
#                 "small": (small_idx, small_idx),
#                 # "med": (med_idx, med_idx),
#                 "large": (large_idx, large_idx),
#                 "all": (all_idx, all_idx)
#             }
#         analyzer = JTrajAnalysis()
#         results = analyzer.analyze_J_traj(tests_to_perform={"Symmetry", "2_cycles", "3_cycles", "4_cycles"},
#                                           traj=best_solutions, topology=topology, connectivity_threshold=None,
#                                           idx_groups=idx_groups)
#         normalize = lambda fitness: (fitness - np.min(fitness))/(np.max(fitness) - np.min(fitness))
#         fitness = np.load(f"experiments\\EvoGloop\\{exp_name}\\fitness_traj.npy")
#         best_fitness = np.max(fitness, axis=1) if "exp_growth" else np.min(fitness, axis=1)
#         results["fitness"] = normalize(best_fitness) if "exp_growth" in exp_name else 1-normalize(best_fitness)
#
#         all_results[exp_name] = results
#         print(exp_name, results)
#     pickle.dump(all_results, open(f"experiments\\analysis", 'wb'))
#
#     exit()
#
#
#     #exp_name="2021_08_27_10_43_14_Shani_fitness_f_type=avg_T_ws=[2000]_tournament_k=2_n_mag_idx=172"
#     for i, exp_name in enumerate(os.listdir(f"experiments\\EvoGloop")):
#         best_sol_analysis(exp_name=exp_name, first_half_idx=None)
#         best_sol_plot(exp_name=exp_name)
#     exit()
#
#
#     dir = "evo_init_cond_search"
#     for j in range(10):
#         J_pop_size = 5
#         n_spins = 2**14
#         topology = np.array(gne.gen_topology(n_spins=n_spins))
#         mag_loops = np.arange(len(topology)//4)
#         Js = gne.create_J_C(J_C=None, same_J_C=False, J_C_sigma=util.DEFAULT_GAMMA / np.sqrt(len(topology)),
#                             n_realizations=J_pop_size, topology=topology)
#         np.save(f"{dir}\\topologies\\topology_{j}.npy", topology)
#         for i in range(J_pop_size):
#             time1 = time.time()
#             J = Js[i]
#             np.save(f"{dir}\\Js\\topology_{j}_J_{i}.npy", J)
#             full_mag_rt = get_lag_times(topology, J, np.arange(len(topology)), n_realizations=900, spins_p=0.5)
#             plot_kill_curve(full_mag_rt, xscale="linear", title=f"topology_{j}_J_{i} Full", save_dir=f"{dir}\\topology_{j}_J_{i}_full_linear")
#             plot_kill_curve(full_mag_rt, xscale="log", title=f"topology_{j}_J_{i} Full", save_dir=f"{dir}\\topology_{j}_J_{i}_full_log")
#             partial_mag_rt = get_lag_times(topology, J, mag_loops, n_realizations=900, spins_p=0.5)
#             plot_kill_curve(partial_mag_rt, xscale="linear", title=f"topology_{j}_J_{i} Partial", save_dir=f"{dir}\\topology_{j}_J_{i}_partial_linear")
#             plot_kill_curve(partial_mag_rt, xscale="log", title=f"topology_{j}_J_{i} Partial", save_dir=f"{dir}\\topology_{j}_J_{i}_partial_log")
#             print(f"topology_{j}_J_{i} completed. Timed at {(time2-time1):.3f} s")

if __name__ == '__main__':
    # topology, J = np.load("topologies_and_Js\\topology_1.npy"), np.load("topologies_and_Js\\topology_1_J_1.npy")
    # lag_times = run_J(topology=topology, J=J, n_realizations_per_J=900, T_ws=util.T_WS, record_mag=False)
    # plot_kill_curves(lag_times)


    from gloop_net_ltee import competition_assay
    def retrieve_mutant(mutant_dict, mutant_idx, ancestor_J):
        cur_idx = mutant_idx
        J = np.copy(ancestor_J)
        trace = [cur_idx]
        while cur_idx != 0:
            cur_idx = mutant_dict[cur_idx]["parent_id"]
            trace = [cur_idx] + trace

        for m in trace[1:]:
            mutated_node = mutant_dict[m]["mutated_node"]
            mutated_connections = mutant_dict[m]["mutated_connections"]
            row_col_mutation = 0 if mutant_dict[m]["row_col_mutation"] == "row" else 1
            if row_col_mutation:
                J[:, mutated_node] = mutated_connections
            else:
                J[mutated_node, :] = mutated_connections
        return J, len(trace)


    topology, ancestor_J = np.load("topologies_and_Js\\topology_1.npy"), np.load(
        "topologies_and_Js\\topology_1_J_1.npy")

    @util.time_func
    def fitness_f(J, mu):
        Ns = competition_assay(Js=[ancestor_J, J], topology=topology, T_w=100, init_size=1000, final_size=50_000,
                               growth_rate=mu)
        return Ns[1] / Ns[0]


    load_from_exp = lambda exp_name, file: pickle.load(
        open(fr"C:\Users\Owner\PycharmProjects\GloopNetEvolution\experiments\LTEE\2022_05_29\{exp_name}\{file}", 'rb'))
    save_to_exp = lambda exp_name, file, to_save: pickle.dump(to_save,
                                                              open(
                                                                  fr"C:\Users\Owner\PycharmProjects\GloopNetEvolution\experiments\LTEE\2022_05_29\{exp_name}\{file}",
                                                                  'wb'))

    for i, exp_name in enumerate(os.listdir(fr"experiments\LTEE\2022_05_29")):
        mu = float(exp_name.split("_")[-2].split("=")[1])
        final_N = load_from_exp(exp_name, "final_N")
        mutant_dict = load_from_exp(exp_name, "mutant_dict")
        best_mut_idx = max(final_N[-1], key=lambda x: [final_N[-1][x]])
        mut_J, gen = retrieve_mutant(mutant_dict, best_mut_idx, ancestor_J)
        print(f"{exp_name=}\n{best_mut_idx=} ({gen=})")
        if best_mut_idx > 0:
            for mutant in mutant_dict:
                mutant_dict[mutant]["fitness"] = fitness_f(mut_J, mu=mu)
            save_to_exp(exp_name, "mutant_dict_", mutant_dict)
            break