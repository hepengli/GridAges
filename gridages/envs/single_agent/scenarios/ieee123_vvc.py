from collections import OrderedDict
from pathlib import Path
import pickle
import numpy as np
from gridages.envs.single_agent.base_env import GridBaseEnv
from gridages.envs.single_agent.scenarios.ieee123_scenario import (
    build_ieee123_devices,
)

#Dataset

def read_data(
        train,
        load_area = "AVA",
        renew_area="NP15",
        price_area="0096WD_7_N001",
):
    
    repo_root = Path(__file__).resolve().parents[4]

    data_path = (
        repo_root
        / "data"
        / "data2023-2024.pkl"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )
    
    with open(data_path, "rb") as file:
        dataset = pickle.load(file)

    return{
        "load":
            dataset[train]["load"][load_area],

        "solar":
            dataset[train]["solar"][renew_area],

        "wind":
            dataset[train]["wind"][renew_area],

        "price":
            dataset[train]["price"][price_area],
    }

#IEEE-123 VVC environment

class IEEE123VVCEnv(GridBaseEnv):

    #Build network

    def _build_net(self):

        net, all_devices, area = build_ieee123_devices(
            task = "vvc"
        )

        self.net = net
        self.area = area

        self.all_devices = all_devices

        #Devices that are VVC controls

        vvc_names = [
            #PV inverter reactive power
            "PV 1",
            "PV 2",
            "PV 3",
            "PV 4",
            "PV 5",
            "PV 6",

            #Wind inverter reactive power
            "WP 1",
            "WP 2",
            "WP 3",
            "WP 4",

            #Voltage regulator taps
            "Regulator 1",
            "Regulator 2",
            "Regulator 3",
            "Regulator 4",

            #Switched capacitor banks
            "SCB 1",
            "SCB 2",
        ]

        self.devices = OrderedDict(
            (name, all_devices[name])
            for name in vvc_names
        )

        #Physical RES active-power ratings
        #45% PV + 35% wind = 80% of 3.49 MW
        #Renewable names are useful later
        #for calculating renewable safety/penalties.

        self.res_names = [
            "PV 1",
            "PV 2",
            "PV 3",
            "PV 4",
            "PV 5",
            "PV 6",

            "WP 1",
            "WP 2",
            "WP 3",
            "WP 4",
        ]

        #Dataset
        self.dataset = read_data(
            self.train,
            load_area="AVA",
            renew_area="NP15",
            price_area="0096WD_7_N001",
        )

        self.last_metrics = {}
        
    # #Push Device statesto pandapower

    def _push_devices_to_net(self):
        
        for name, dev in self.devices.items():
            table, idx = self._pp_refs[name]

            cls = dev.__class__.__name__


    #         #Renewable

            if cls == "RES":
                dev.update_state()

                self.net.sgen.at[
                    idx,
                    "p_mw"
                ]=dev.state.P

                self.net.sgen.at[
                    idx,
                    "q_mvar"
                ]=dev.state.Q

    #         #SCB
            elif cls == "Shunt":
                dev.update_state()

                #Action itself is the capacitor step.
                step = int(
                    dev.action.d[0]
                )

                #Pandapower already stores
                #q_mvar per step.

                self.net.shunt.at[
                    idx,
                    "q_mvar"
                ] = float(
                    dev.q_mvar
                )

                self.net.shunt.at[
                    idx,
                    "step"
                ] = step

    #         #Regulator

            elif cls == "Transformer":

                dev.update_state()

                self.net.trafo.at[
                    idx,
                    "tap_pos"
                ] = int(
                    dev.state.tap_position
                )

    #         #Grid itself requires no push here.

    #VVC reward and safety

    def _reward_and_safety(self):

        #Power flow convergence

        if not self.net.converged:

            reward = {
                "vvc": -100.0
            }

            safety = {
                "convergence": 1.0
            }

            return reward, safety
        
        #Line active power loss

        line_loss_mw = float(
            self.net.res_line["pl_mw"].sum()
        )

        #Voltage Violation (0.95 <= V <= 1.05 pu)

        vm = self.net.res_bus[
            "vm_pu"
        ].to_numpy()

        undervoltage = np.maximum(
            0.95 - vm,
            0.0,
        ).sum()

        overvoltage = np.maximum(
            vm - 1.05,
            0.0,
        ).sum()

        voltage_violation = float(
            undervoltage
            + overvoltage
        )

        #Requested inverter- capability/ Q violation
        #GridAges RES safety checks whether:
        #P^2 + Q^2 <= S^2
        #if not, this captures violation of inverter apparent-power capability

        q_violation = 0.0

        for name in self.res_names:

            q_violation += float(
                self.devices[name].safety
            )


        #Line overload
        #Safety metric but not included in VVC reward

        loading = self.net.res_line[
            "loading_percent"
        ].to_numpy()

        line_overload = float(
            np.maximum(
                loading - 100.0,
                0.0,
            ).sum()
            /100.0
        )

        #VVC reward
        # r = -(100 P_loss + Q_violation + V_violation)

        reward_value = -(
            100.0 * line_loss_mw + q_violation + voltage_violation
        )

        reward = {
            "vvc": float(
                reward_value
            )
        }

        safety = {
            "q_violation":
                q_violation,

            "voltage_violation":
                voltage_violation,

            "line_overload":
                line_overload,
        }

        #Save Diagnostics for logging and testing

        self.last_metrics = {
            "line_loss_mw":
                line_loss_mw,

            "q_violation":
                q_violation,

            "voltage_violation":
                voltage_violation,

            "line_overload":
                line_overload,

            "vmin":
                float(vm.min()),

            "vmax":
                float(vm.max()),

            "max_line_loading":
                float(loading.max()),

            "reward":
                float(reward_value),
        }

        self.reward = reward
        self.safety = safety

        return reward, safety
        
    #Reset
    def reset(
            self,
            *,
            seed = None,
            options = None,
    ):
        
        #Let GridAges perform its normal reset first
        super().reset(
            seed=seed,
            options = options,
        )

        #Neutral initial VVC action
        #Regulator category corresponding to tap = 0:
        #Reg1: -2+2=0
        #Reg2: -16+16=0
        #Reg3: -16+16=0
        #Reg4: -2+2=0

        #SCBs = step 0

        #Neutral VVC operating point
        neutral_action ={
            "continuous":
                np.zeros(
                    10,
                    dtype=np.float32,
                ),

            "discrete":
                np.array(
                    [
                        2, #Regulator 1 -> tap 0
                        16, #Regulator 2 -> tap 0
                        16, #Regulator 3 -> tap 0
                        2, #Regulator 4 -> tap 0
                        0, #SCB 1 off
                        0, #SCB 2 off
                    ],
                    dtype=np.int64,
                ),
        }

        #Use GridAges inherited mapping
        self._set_action(
            neutral_action
        )

        #Use our VVC-specific device-to-network push
        self._push_devices_to_net()

        #Solve the neutral operating point
        converged = self._solve_pf()

        #Update GridAges device safety.cost states
        self._update_device_cost_safety(
            converged
        )

        #Recalculate VVC reward/safety
        self._reward_and_safety()

        obs = self._get_obs()

        info = self._collect_info(
            converged
        )

        return obs, info





        

