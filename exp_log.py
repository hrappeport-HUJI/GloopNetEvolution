import csv
import datetime
import os
import util




class ExpLog(dict):
    """
    Object for logging experiment related data.
    """
    def __init__(self, default_args, qwargs, save_log=True, ignore_args=False):
        super().__init__()
        if ignore_args: return

        if default_args:
            print(f"The following experiment arguments were not passed and were set to their default value:")
            print(", ".join([f"{arg_name} (set to {default_args[arg_name]})" for arg_name in default_args]))

        if qwargs:
            print(f"The following arguments were passed but not recognized (discarded):")
            print(", ".join([f"{arg_name} (with value {qwargs[arg_name]})" for arg_name in qwargs]))
        if save_log: self.save()

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def set_optional_arg(qwargs, default_args, arg_name, default_val):
        if arg_name in qwargs:
            return qwargs.pop(arg_name)
        default_args[arg_name] = default_val
        return default_val

    @staticmethod
    def set_required_arg(qwargs, arg_name):
        if arg_name in qwargs:
            return qwargs.pop(arg_name)
        raise Exception(f"Required argument {arg_name} not passed")

    @staticmethod
    def _get_date_str():
        now = datetime.datetime.now()
        str0 = lambda s: f"0{s}" if s<10 else f"{s}"
        return f"{now.year}_{str0(now.month)}_{str0(now.day)}_{str0(now.hour)}_{str0(now.minute)}_{str0(now.second)}"

    def save(self, path=None):
        path = path or os.path.join(self.dir, "log")
        with open(f"{path}.csv", "w") as f:
            writer = csv.writer(f)
            for key, val in self.items():
                writer.writerow([key, val])
        print(f"Experiment-log for experiment {self.exp_name} saved to {path}")

    @staticmethod
    def load(exp_name):  # TODO test
        path = os.path.join(util.PROJECT_CWD, "experiments", exp_name, "log")
        exp_log = GloopNetExpLog(ignore_args=True)
        with open(f"{path}.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row: exp_log[row[0]] = row[1]
        return exp_log


class GloopNetExpLog(ExpLog):
    def __init__(self, save_log=True, ignore_args=False, **qwargs):
        if ignore_args: return
        default_args = {}
        set_required_arg = lambda arg_name: ExpLog.set_required_arg(qwargs, arg_name)
        set_optional_arg = lambda arg_name, default_val: ExpLog.set_optional_arg(qwargs, default_args, arg_name, default_val)

        # Required arguments
        self.topology = set_required_arg("topology")
        self.n_spins = set_required_arg("n_spins")
        self.sim_len = set_required_arg("sim_len")
        self.n_realizations = set_required_arg("n_realizations")
        self.H = set_required_arg("H")

        # Optional arguments
        self.device = set_optional_arg("device", default_val="cpu")
        if "exp_name" in qwargs: qwargs["exp_name"] = f"{ExpLog._get_date_str()}_{qwargs['exp_name']}"
        self.exp_name = set_optional_arg("exp_name", default_val=GloopNetExpLog._get_date_str())
        # self.n_batches = set_optional_arg("n_batches", default_val=1)  # TODO
        self.dir = os.path.join(util.PROJECT_CWD, "experiments", self.exp_name)
        if not os.path.isdir(self.dir) and save_log: os.makedirs(self.dir)

        self.gamma = set_optional_arg("gamma", default_val=None)
        self.J_C = set_optional_arg("J_C", default_val=None)  # A value other than None overrides J_C_sigma
        self.same_J_C = set_optional_arg("same_J_C", default_val=False)
        # self.J_C_sigma = set_optional_arg("J_C_sigma", default_val=None)

        self.spins = set_optional_arg("spins", default_val=None)  # A value other than None overrides spins_p
        self.same_init_spins = set_optional_arg("same_init_spins", default_val=True)
        self.spins_p = set_optional_arg("spins_p", default_val=0.5)

        super().__init__(default_args, qwargs, save_log, ignore_args)


class EvoGloopExpLog(ExpLog):
    def __init__(self, save_log=True, ignore_args=False, **qwargs):
        if ignore_args: return
        default_args = {}
        set_required_arg = lambda arg_name: ExpLog.set_required_arg(qwargs, arg_name)
        set_optional_arg = lambda arg_name, default_val: ExpLog.set_optional_arg(qwargs, default_args, arg_name, default_val)

        # Required arguments

        self.gen_init_pop_f = set_required_arg("gen_init_pop_f")
        self.fitness_f = set_required_arg("fitness_f")
        self.selection_f = set_required_arg("selection_f")
        self.mutation_f = set_required_arg("mutation_f")
        self.mut_params_mutation_f = set_required_arg("mut_params_mutation_f")
        self.termination_cond = set_required_arg("termination_cond")

        # Optional arguments
        if "exp_name" in qwargs: qwargs["exp_name"] = f"{ExpLog._get_date_str()}_{qwargs['exp_name']}"
        self.exp_name = set_optional_arg("exp_name", default_val=GloopNetExpLog._get_date_str())

        self.device = set_optional_arg("device", default_val="cuda:0")

        self.dir = os.path.join(util.PROJECT_CWD, "experiments", "EvoGloop", self.exp_name)
        if not os.path.isdir(self.dir) and save_log: os.makedirs(self.dir)

        super().__init__(default_args, qwargs, save_log, ignore_args)



class LTEEExpLog(ExpLog):
    def __init__(self, save_log=False, create_dir=True, ignore_args=False, **qwargs):
        if ignore_args: return
        default_args = {}
        set_required_arg = lambda arg_name: ExpLog.set_required_arg(qwargs, arg_name)
        set_optional_arg = lambda arg_name, default_val: ExpLog.set_optional_arg(qwargs, default_args, arg_name, default_val)


        # Required arguments
        self.topology = set_required_arg("topology")
        self.C_idx = set_required_arg("C_idx")
        self.FC_idx = set_required_arg("FC_idx")
        self.H = set_required_arg("H")
        self.lag_start_time = set_required_arg("lag_start_time")
        self.gen_init_Js = set_required_arg("gen_init_Js")
        self.gen_init_spins = set_required_arg("gen_init_spins")
        self.gen_init_mut_params = set_required_arg("gen_init_mut_params")
        self.mutator = set_required_arg("mutator")
        self.termination_cond = set_required_arg("termination_cond")
        self.growth_rate = set_required_arg("growth_rate")
        self.bottleneck_size = set_required_arg("bottleneck_size")
        self.stationary_size = set_required_arg("stationary_size")
        self.logger = set_required_arg("logger")



        # Optional arguments
        self.max_time = set_optional_arg("max_time", default_val=None)

        if "exp_name" in qwargs: qwargs["exp_name"] = f"{ExpLog._get_date_str()}_{qwargs['exp_name']}"
        self.exp_name = set_optional_arg("exp_name", default_val=GloopNetExpLog._get_date_str())

        self.dir = os.path.join(util.PROJECT_CWD, "experiments", "LTEE", self.exp_name)
        if not os.path.isdir(self.dir) and create_dir: os.makedirs(self.dir)
        self.logger.dir = self.dir

        super().__init__(default_args, qwargs, save_log, ignore_args)


