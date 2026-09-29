import pickle
from pathlib import Path

import numpy as np

from gymnasium.spaces import(
    Box,
    Discrete,
    Dict as SpaceDict,
)

from pandapower.toolbox import(
    get_element_index
)

from gridages.envs.single_agent.base_env import(
    GridBaseEnv
)
from gridages.envs.single_agent.scenarios.ieee13_scenario import(
    build_ieee13_devices
)

#Time series data

def read_data(train):
    
    repo_root = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    data_path = (
        repo_root
        /"data"
        /"data2023-2024.pkl"
    )

    with open(
        data_path,
        "rb"
    ) as file:
        
        dataset = pickle.load(
            file
        )

    return {
        "load":
            dataset[train]
            ["load"]
            ["AVA"],

        "solar":
            dataset[train]
            ["solar"]
            ["NP15"],
        
        "wind":
            dataset[train]
            ["wind"]
            ["NP15"],

        "price":
            dataset[train]
            ["price"]
            ["0096WD_7_N001"],
    }

#IEEE-13 VVC Environment
class IEEE13VVCEnv(
    GridBaseEnv
):
    
    def _build_net(self):

        #Shared Physical scenario

        net, devices, area = build_ieee13_devices(
            task="vvc"
        )

        self.net = net
        self.devices = devices
        self.area = area

        #Timeseries data

        self.dataset = read_data(
            self.train
        )
        
        #-------------------------------
        #Reward Coefficients
        #
        #omega_1 = 100: line loss/ power loss
        #omega_2 = 1: Q-limit violation
        #omega_3 = 1: voltage violation

        self.omega_loss = float(
            self.cfg.get(
                "omega_loss",
                100.0
            )
        )

        self.omega_q = float(
            self.cfg.get(
                "omega_q",
                1.0
            )
        )
        
        self.omega_v = float(
            self.cfg.get(
                "omega_v",
                1.0
            )
        )
        self.last_vvc_terms={}

    def _build_action_space(self):
        pv = self.devices["PV1"]
        wt = self.devices["WT1"]

        #Continuous actions: [Q_PV, Q_WT]

        continuous_low = np.array(
            [
                pv.min_q_mvar,
                wt.min_q_mvar,
            ],
            dtype=np.float32
        )
        
        continuous_high = np.array(
            [
                pv.max_q_mvar,
                wt.max_q_mvar,
            ],
            dtype=np.float32
        )

        #Discrete transformer action
        substation = self.devices[
            "Substation"
        ]

        number_of_taps = (
            substation.tap_max
            - substation.tap_min
            +1
        )

        return SpaceDict({

            "continuous": Box(
                low = continuous_low,
                high=continuous_high,
                dtype=np.float32,
            ),

            "discrete": Discrete(
                number_of_taps
            ),
        })

    def _set_action(self, action):
        
        q_action = np.asarray(
            action["continuous"],
            dtype=np.float32
        )

        tap_action = int(
            action["discrete"]
        )

        #PV inverter Q

        pv = self.devices["PV1"]

        pv.action.c[:] = [
            q_action[0]
        ]

        #wind inverter Q

        wt = self.devices["WT1"]

        wt.action.c[:] = [
            q_action[1]
        ]

        #BESS is present in network but not controlled by VVC
        #P_ESS = 0

        ess = self.devices["ESS1"]
        ess.action.c[:]=0.0

        #Dispatchable DG is present in network but not controlled by VVC.
        #P_DG = 0

        dg = self.devices["DG1"]

        dg.action.c[:] = 0.0

        #substation tap

        substation = self.devices[
            "Substation"
        ]

        substation.action.d[:]=[
            tap_action
        ]

    def _reward_and_safety(self):

        #Power flow failure

        if not self.net["converged"]:

            reward ={
                "vvc":-100.0
            }

            safety = {
                "convergence": 1.0
            }

            self.last_vvc_terms={
                "line_loss_mw": 0.0,
                "q_violation": 0.0,
                "voltage_violation": 0.0,
                "line_overload": 0.0,
            }

            self.reward = reward
            self.safety = safety

            return reward, safety
        
        #Bus voltages

        bus_ids = get_element_index(
            self.net,
            "bus",
            self.area,
            False
        )

        vm = (
            self.net
            .res_bus
            .loc[
                bus_ids,
                "vm_pu"
            ]
            .values
        )

        #Voltage violation
        #sum[max(V-1.05, 0) + max(0.95-V, 0)]

        over_voltage = np.maximum(
            vm-1.05, 0.0
        )

        under_voltage = np.maximum(
            0.95 - vm,
            0.0
        )

        voltage_violation = float(
            (
                over_voltage
                +
                under_voltage
            ).sum()
        )

        #Line losses

        line_ids = get_element_index(
            self.net,
            "line",
            self.area,
            False
        )

        line_loss_mw = float(
            self.net
            .res_line
            .loc[
                line_ids,
                "pl_mw"
            ]
            .sum()
        )

        #Renewable inverter Q violations
        # |Q| <= sqrt(S^2-P^2)

        q_violation = 0.0

        for name in [
            "PV1",
            "WT1",
        ]:
            
            dev = self.devices[
                name
            ]
            p=float(
                dev.state.P
            )
            q=float(
                dev.state.Q
            )
            s_rating = float(
                dev.sn_mva
            )
            q_max = np.sqrt(
                max(
                    s_rating ** 2
                    -
                    p ** 2,
                    0.0
                )
            )

            q_violation +=max(
                0.0, q-q_max
            )

            q_violation += max(
                0.0,
                -q_max - q
            )

        #VVC reward
        # r = 
        # - w1 * line loss
        # -w2 * Q violation
        # - w3 * voltage violation

        reward_value = -(
            self.omega_loss
            * line_loss_mw

            +

            self.omega_q
            * q_violation

            +

            self.omega_v
            * voltage_violation
        )

        #Line-overloading safety

        loading = (
            self.net.res_line.loc[
                line_ids,
                "loading_percent"
            ]
            .values
        )

        line_overload = float(
            np.maximum(
                loading/ 100.0
                -1.0,
                0.0
            ).sum()
        )

        reward = {
            "vvc":
                float(
                    reward_value
                )
        }

        safety = {

            "voltage_violation":
            voltage_violation,

            "q_violation":
            float(
                q_violation
            ),

            "line_overload":
            line_overload
        }

        #Save individual values for debugging/analysis

        self.last_vvc_terms={
            "line_loss_mw":
            line_loss_mw,

            "q_violation":
            float(
                q_violation
            ),

            "voltage_violation":
            voltage_violation,

            "line_overload":
            line_overload,
        }

        self.reward = reward
        self.safety = safety

        return reward, safety

