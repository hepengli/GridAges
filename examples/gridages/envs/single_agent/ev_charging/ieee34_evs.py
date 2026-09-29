import os
import pickle
from collections import OrderedDict
from os.path import abspath, dirname

import numpy as np
import pandapower as pp

from gridages.envs.single_agent.base_env import GridBaseEnv
from gridages.networks.ieee34 import IEEE34Bus
from gridages.devices import EV, RES, Grid, Transformer


def _hourly_to_half_hourly(x):
    """Convert an hourly series to 30-minute resolution by midpoint interpolation."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.empty(x.size * 2, dtype=np.float32)
    y[0::2] = x
    y[1:-1:2] = 0.5 * (x[:-1] + x[1:])
    y[-1] = x[-1]
    return y


def read_data(train, load_area, renew_area, price_area, half_hour=True):
    root = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
    data_dir = os.path.join(root, "data", "data2023-2024.pkl")

    with open(data_dir, "rb") as f:
        dataset = pickle.load(f)

    data = {
        "load": np.asarray(dataset[train]["load"][load_area], dtype=np.float32),
        "solar": np.asarray(dataset[train]["solar"][renew_area], dtype=np.float32),
        "wind": np.asarray(dataset[train]["wind"][renew_area], dtype=np.float32),
        "price": np.asarray(dataset[train]["price"][price_area], dtype=np.float32),
    }

    if half_hour:
        data = {k: _hourly_to_half_hourly(v) for k, v in data.items()}

    return data


class IEEE34Env(GridBaseEnv):
    """IEEE 34-bus EV smart-charging environment.

    The environment intentionally relies on GridBaseEnv for action handling,
    observation construction, stepping, reset, and pandapower synchronization.
    Only the network/scenario and the study-specific reward are customized here.

    Configuration
    -------------
    ev_level : {"low", "medium", "high"}
        Low    = 28 EVs  (18 residential, 8 office, 2 commercial)
        Medium = 56 EVs  (36 residential, 16 office, 4 commercial)
        High   = 84 EVs  (54 residential, 24 office, 6 commercial)

    control_mode : {"rl", "uncontrolled"}
        rl           -> EV charging power is supplied by the RL action.
        uncontrolled -> each connected EV charges immediately at its feasible
                        maximum rate until reaching its target SOC.
    """

    DT_HOURS = 0.5
    DAILY_STEPS = 48
    # Level-2 charging power by usage type.
    # Residential charging is kept moderate; office/commercial chargers use
    # high-power Level-2 charging.
    CHARGER_MW = {
        "residential": 0.0072,  # 7.2 kW
        "office": 0.0192,       # 19.2 kW
        "commercial": 0.0192,   # 19.2 kW
    }

    EV_COUNTS = {
        "low": {"residential": 18, "office": 8, "commercial": 2},
        "medium": {"residential": 36, "office": 16, "commercial": 4},
        "high": {"residential": 54, "office": 24, "commercial": 6},
    }

    EV_BUSES = {
        "residential": ["Bus 826", "Bus 836", "Bus 848"],
        "office": ["Bus 822", "Bus 844"],
        "commercial": ["Bus 840"],
    }

    # Deterministic mix of common passenger-EV battery sizes.
    BATTERY_MWH = (0.050, 0.065, 0.085)

    def _build_net(self):
        self.area = "MG"

        # One episode represents 08:00 today -> 08:00 the next day.
        # GridBaseEnv keeps dataset index, episode step, and real-world time
        # separate through data_idx, step_idx, and time.
        self.episode_length = self.DAILY_STEPS
        self.dt = self.DT_HOURS
        self.start_hour = 8.0

        self.net = IEEE34Bus(self.area)

        ev_level = str(self.cfg.get("ev_level", "medium")).lower()
        if ev_level not in self.EV_COUNTS:
            raise ValueError(
                f"Unknown ev_level={ev_level!r}; choose from {tuple(self.EV_COUNTS)}"
            )

        control_mode = str(self.cfg.get("control_mode", "rl")).lower()
        if control_mode not in ("rl", "uncontrolled"):
            raise ValueError("control_mode must be 'rl' or 'uncontrolled'")

        external_control = control_mode == "rl"
        self.control_mode = control_mode
        self.ev_level = ev_level

        devices = []

        # EVs: geographically distributed across the feeder.  Bus 890 is
        # intentionally avoided because it sits behind the small downstream
        # transformer and would make that local constraint dominate the study.
        for usage_type, n_ev in self.EV_COUNTS[ev_level].items():
            buses = self.EV_BUSES[usage_type]

            for i in range(n_ev):
                capacity = self.BATTERY_MWH[i % len(self.BATTERY_MWH)]
                bus = buses[i % len(buses)]
                name = f"{usage_type.upper()}_EV_{i + 1:03d}"

                ev = EV(
                    name=name,
                    bus=bus,
                    min_p_mw=0.0,              # charging only; no V2G
                    max_p_mw=self.CHARGER_MW[usage_type],
                    capacity=capacity,
                    min_e_mwh=0.20 * capacity,
                    max_e_mwh=0.80 * capacity,
                    init_soc=0.50,              # fraction, not MWh
                    dt=self.DT_HOURS,
                    usage_type=usage_type,
                    daily_steps=self.DAILY_STEPS,
                    external_control=external_control,
                )
                devices.append((ev.name, ev))

        # PV is retained as an environmental condition.  It affects net feeder
        # demand naturally through the power flow; it is not directly rewarded.
        pv_specs = [
            ("PV_RESIDENTIAL", "Bus 826", 0.10),
            ("PV_OFFICE", "Bus 822", 0.10),
            ("PV_COMMERCIAL", "Bus 840", 0.10),
        ]
        for name, bus, rating in pv_specs:
            pv = RES(name, bus=bus, sn_mva=rating, source="solar")
            devices.append((pv.name, pv))

        # External grid rating follows the regulator scale in the GridAges
        # IEEE-34 model.
        grid = Grid(
            "Grid",
            bus="Bus 800",
            sn_mva=2.5,
            sell_discount=0.9,
            dt=self.DT_HOURS,
        )
        devices.append((grid.name, grid))

        # Register existing IEEE-34 transformers so the base environment can
        # evaluate their loading safety.
        for _, row in self.net.trafo.iterrows():
            name = row["name"][len(self.area) + 1 :]
            trafo = Transformer(
                name=name,
                sn_mva=row["sn_mva"],
                dt=self.DT_HOURS,
            )
            devices.append((trafo.name, trafo))

        self.devices = OrderedDict(devices)

        # GridBaseEnv expects all four keys.  No wind device is installed, but
        # retaining the wind series keeps the environment compatible with the
        # generic data-scaling code.
        self.dataset = read_data(
            self.train,
            load_area="AZPS",
            renew_area="SP15",
            price_area="0096WD_7_N001",
            half_hour=True,
        )

        # Reward bookkeeping.  No reset override is required: _reward_and_safety
        # clears these whenever a new 48-step episode begins.
        self._grid_p_history = []
        self._previous_pvr = 0.0

    # ------------------------------------------------------------------
    # Study-specific reward and safety
    # ------------------------------------------------------------------
    @staticmethod
    def _pvr(power_history):
        """Peak-to-valley range normalized by mean feeder demand."""
        p = np.asarray(power_history, dtype=np.float64)
        if p.size <= 1:
            return 0.0

        p_mean = max(np.mean(np.abs(p)), 1e-6)
        return (np.max(p) - np.min(p)) / p_mean

    def _reward_and_safety(self):
        if not self.net["converged"]:
            reward = {"convergence": -100.0}
            safety = {"convergence": 1.0}

            self.reward = reward
            self.safety = safety
            return reward, safety

        # --------------------------------------------------------------
        # Episode bookkeeping
        # --------------------------------------------------------------
        # Reward history is episode-relative and independent of data_idx.
        if self.step_idx == 0:
            self._grid_p_history = []
            self._previous_pvr = 0.0

        # Positive p_mw at the external grid means feeder import.
        grid_p_mw = self.net.res_ext_grid.iloc[0]["p_mw"]
        self._grid_p_history.append(grid_p_mw)

        current_pvr = self._pvr(self._grid_p_history)
        delta_pvr = current_pvr - self._previous_pvr
        self._previous_pvr = current_pvr

        # ==============================================================
        # 1) REWARD: load-shape objective
        # ==============================================================
        pvr_weight = self.cfg.get("pvr_weight", 1.0)
        terminal_pvr_weight = self.cfg.get("terminal_pvr_weight", 2.0)

        # Dense signal:
        # reward the agent for reducing the running PVR and penalize
        # actions that increase it.
        reward = {
            "pvr_delta": -pvr_weight * delta_pvr
        }

        next_step = self.step_idx + 1
        # Final daily objective.
        # _reward_and_safety() is evaluated before GridBaseEnv increments
        # step_idx, so step_idx + 1 is the step that has just completed.
        is_last_step = next_step >= self.episode_length
        if is_last_step:
            reward["pvr_terminal"] = -terminal_pvr_weight * current_pvr

        # ==============================================================
        # 2) REWARD: EV charging-service requirement
        # ==============================================================
        # An EV may be inside its physical SOC limits but still leave
        # without enough energy.
        #
        # Example:
        #     min_soc = 0.20
        #     max_soc = 0.80
        #     departure SOC = 0.55
        #
        # SOC=0.55 is physically valid, so it is NOT a safety violation.
        # However, it fails the charging-service requirement and therefore
        # receives a negative reward.
        #
        # Using unmet energy in MWh rather than raw SOC makes penalties
        # comparable across vehicles with different battery capacities.
        unmet_weight = self.cfg.get("unmet_weight", 100.0)
        unmet_mwh = 0.0


        for dev in self.devices.values():
            if not isinstance(dev, EV):
                continue

            # EV arrival/departure times are expressed in real-world
            # episode hours:
            #
            #   16.0 -> 4 PM today
            #   24.0 -> midnight
            #   32.0 -> 8 AM next day
            #
            # The EV is allowed to charge during the current interval.
            # We evaluate its service requirement at the end of the
            # interval if its departure occurs there.
            leaves_after_this_step = (
                dev.connected
                and self.step_idx < dev.depart_time <= next_step + 1e-9
            )

            if leaves_after_this_step:
                unmet_energy = max(
                    (dev.max_soc - dev.state.soc)
                    * dev.capacity,
                    0.0,
                )

                unmet_mwh += unmet_energy

        if unmet_mwh > 0.0:
            reward["unmet_energy"] = -unmet_weight * unmet_mwh

        # ==============================================================
        # 3) Optional economic reward
        # ==============================================================
        # Default = 0 so the primary scientific question remains
        # peak-demand reduction rather than electricity-cost optimization.
        price_weight = self.cfg.get("price_weight", 0.0)

        if price_weight > 0.0:
            grid = self.devices.get("Grid")

            if grid is not None:
                reward["energy_cost"] = -price_weight * grid.cost

        # ==============================================================
        # 4) SAFETY: device and network constraints
        # ==============================================================
        # IMPORTANT:
        #
        # RL actions are NOT corrected with EV.feasible_action().
        #
        # If the RL controller overcharges an EV or otherwise drives SOC
        # outside the physical limits, ESS/EV update_cost_safety() detects
        # that violation through soc_bounds_penalty().
        #
        # This allows the agent to explore invalid SOC behavior and learn
        # from the resulting safety penalty rather than having the
        # environment silently modify its action.

        bus_ids = pp.get_element_index(
            self.net,
            "bus",
            self.area,
            False,
        )
        vm = self.net.res_bus.loc[bus_ids].vm_pu.values
        overvoltage = np.maximum(vm - 1.05, 0.0, ).sum()
        undervoltage = np.maximum(0.95 - vm, 0.0,).sum()

        line_ids = pp.get_element_index(
            self.net,
            "line",
            self.area,
            False,
        )
        line_loading = self.net.res_line.loc[line_ids].loading_percent.values
        overloading = np.maximum(line_loading - 100.0, 0.0,).sum() * 0.01

        # Device-local safety includes:
        #
        #   EV/ESS SOC violations
        #   transformer loading
        #   device rating violations
        #
        # For EVs specifically, this is where overcharging is penalized.
        safety = {name: dev.safety for name, dev in self.devices.items()}

        safety.update(
            {
                "line_overloading": overloading,
                "overvoltage": overvoltage,
                "undervoltage": undervoltage,
            }
        )

        self.reward = reward
        self.safety = safety

        return reward, safety

if __name__ == "__main__":
    from gridages.envs.single_agent.ev_charging.ieee34_evs import IEEE34Env
    env = IEEE34Env(
        env_config={
            "train": False,
            "ev_level": "high",
            "control_mode": "uncontrolled",
            "episode_length": 48,
            "dt": 0.5,
            "start_hour": 8.0,
            "store_info": ["bus_voltage", "line_loading", "operating_cost"],
            "safety_scale": 1.0,
            "max_penalty": 100.0,
        }
    )

    obs, info = env.reset(seed=1)
    done = False
    total_reward = 0.0
    total_safety = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print("step:", env.step_idx, "reward:", reward, "safety:", info["s"])
        total_reward += reward
        total_safety += info["s"]
        done = terminated or truncated
        # print(env.devices["Grid"].state.P, env.devices["Grid"].state.price, env.devices["Grid"].cost, env.devices["Grid"].safety)
        print(info["bus_voltage_viol_count"], info["line_loading_mean"], info["operating_cost"])

    print("episode reward:", total_reward)
    print("final PVR:", env._pvr(env._grid_p_history))
    print("total safety penalty:", total_safety)
