import os
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from gridages.envs.single_agent.optimal_power_flow.ieee13_mg import IEEE13Env

LOG_DIR = "./logs/ppo"
TB_LOG = os.path.join(LOG_DIR, "tb")
SAVE_DIR = "./models/ppo"

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
SEED = 42
N_ENVS = 1

# If your env supports data_path and you want to explicitly point at your dataset:
DATA_PATH: Optional[str] = None  # e.g. "/Users/hepeng.li/Documents/code/python/GridAges/data/data2023-2024.pkl"

# These must match your GridBaseEnv._collect_info implementation.
STORE_INFO = ["operating_cost", "bus_voltage", "line_loading"]
STORE_SUMMARIES = True   # MUST be True to get scalar keys for TensorBoard
STORE_ARRAYS = False     # Arrays are not plotted in TB; set True only if you also save NPZ


class GridScalarTBCallback(BaseCallback):
    """
    Logs scalar info keys from env -> TensorBoard.
    This logs both step-wise values and episode means.
    Also prints available info keys periodically for debugging.
    """
    def __init__(self, keys_to_log: List[str], debug_every: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.keys = keys_to_log
        self.debug_every = debug_every
        self.ep_buf: Dict[str, List[float]] = {k: [] for k in keys_to_log}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if infos and (self.num_timesteps % self.debug_every == 0):
            print("[debug] info keys:", sorted(list(infos[0].keys())))

        for info in infos:
            for k in self.keys:
                v = info.get(k, None)
                if isinstance(v, (int, float, np.number)):
                    fv = float(v)
                    # step-wise logging (gives you curves immediately)
                    self.logger.record(f"grid/{k}", fv)
                    # buffer for episode mean
                    self.ep_buf[k].append(fv)

        # episode ended -> dump episode means
        if np.any(dones):
            for k, vals in self.ep_buf.items():
                if vals:
                    self.logger.record(f"grid/{k}_ep_mean", float(np.mean(vals)))
            self.ep_buf = {k: [] for k in self.keys}

        return True


def make_env(seed: int):
    def _init():
        env_config = {
            "load_scale": 1.0,
            "reward_scale": 1.0,
            "safety_scale": 10000.0,
            "max_penalty": 10000.0,
            "train": True,

            # ---- IMPORTANT: enable info + summaries ----
            "store_info": STORE_INFO,
            "store_summaries": STORE_SUMMARIES,
            "store_arrays": STORE_ARRAYS,
        }
        if DATA_PATH is not None:
            env_config["data_path"] = DATA_PATH

        env = IEEE13Env(env_config=env_config)
        env = Monitor(env)

        # seed
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)

        return env

    return _init


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_LOG, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Train env(s)
    vec_env = DummyVecEnv([make_env(SEED + i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    # Eval env (single)
    eval_env = DummyVecEnv([make_env(SEED + 10_000)])
    eval_env = VecMonitor(eval_env)

    # Model (DO NOT override logger; this ensures TB writes to TB_LOG)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        tensorboard_log=TB_LOG,
        seed=SEED,
        verbose=1,
        device="auto",
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ // max(N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=EVAL_FREQ // max(N_ENVS, 1),
        save_path=SAVE_DIR,
        name_prefix="ppo_ckpt",
    )

    # These keys must exist in info (emitted by GridBaseEnv._collect_info when store_summaries=True)
    tb_keys = [
        "operating_cost",
        "bus_voltage_mean",
        "bus_voltage_min",
        "bus_voltage_max",
        "bus_voltage_viol_count",
        "line_loading_mean",
        "line_loading_max",
        "line_over_100_count",
    ]
    grid_tb = GridScalarTBCallback(keys_to_log=tb_keys, debug_every=2000)

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, ckpt_callback, grid_tb],
        progress_bar=True,
    )

    print(f"[OK] Training complete. TensorBoard logs in: {TB_LOG}")
    print(f"[TIP] Run: tensorboard --logdir {TB_LOG}")


if __name__ == "__main__":
    main()