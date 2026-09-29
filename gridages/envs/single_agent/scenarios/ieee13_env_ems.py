import pickle
from pathlib import Path

import numpy as np
import pandapower as pp
from pandapower.toolbox import get_element_index

from gridages.envs.single_agent.base_env import GridBaseEnv

from gridages.envs.single_agent.scenarios.ieee13_scenario import (
    build_ieee13_devices
)

def read_data(train):
    repo_root = Path(__file__).resolve().parents[4]

    data_path = (
        repo_root/
        "data"/
        "data2023-2024.pkl"
    )

    with open(data_path, "rb") as file:
        dataset = pickle.load(file)

    return {
        "load": dataset[train]["load"]["AVA"],
        "solar": dataset[train]["solar"]["NP15"],
        "wind": dataset[train]["wind"]["NP15"],
        "price": dataset[train]["price"]["0096WD_7_N001"],
    }

class IEEE13ScenarioEnv(GridBaseEnv):
    def _build_net(self):

        #Physical scenario
        net, devices, area = (
            build_ieee13_devices()
        )
        self.net = net
        self.devices = devices
        self.area = area

        #Time-series data
        self.dataset = read_data(
            self.train
        )

    def _reward_and_safety(self):
        
        #Power Flow converged
        if self.net["converged"]:

            #Device operating reward

            reward = { 
                name: -float(dev.cost)
                for name, dev
                in self.devices.items()
            }

            safety={
                name: float(dev.safety)
                for name, dev
                in self.devices.items()
            }

            #Network voltage safety

            bus_ids = get_element_index(
                self.net,
                "bus",
                self.area,
                False
            )

            vm = (
                self.net
                .res_bus
                .loc[bus_ids]
                .vm_pu
                .values
            )

            undervoltage = (
                np.maximum(
                    0.95 - vm,
                    0.0
                ).sum()
            )

            overvoltage = (
                np.maximum(
                    vm - 1.05,
                    0.0
                ).sum()
            )

            #Line Loading Safety

            line_ids = get_element_index(
                self.net,
                "line",
                self.area,
                False
            )

            loading = (
                self.net
                .res_line
                .loc[line_ids]
                .loading_percent
                .values
            )

            overloading = (
                np.maximum(
                    loading - 100.0,
                    0.0
                ).sum()
                *0.01
            )

            safety.update({
                "undervoltage":
                float(undervoltage),

                "overvoltage":
                float(overvoltage),

                "overloading":
                float(overloading),
            })

        #Power flow failed
        else:

            reward = {
                "convergence": -100.0
            }

            safety = {
                "convergence": 1.0
            }

        return reward, safety