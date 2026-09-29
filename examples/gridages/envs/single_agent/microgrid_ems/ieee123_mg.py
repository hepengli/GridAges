import os, pickle
import numpy as np
import pandapower as pp

from os.path import dirname, abspath
from collections import OrderedDict

from gridages.envs.single_agent.base_env import GridBaseEnv
from gridages.networks.ieee13 import IEEE13Bus
from gridages.devices import *


def read_data(train, load_area, renew_area, price_area):
    dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
    data_dir = os.path.join(dir, 'data', 'data2023-2024.pkl')
    with open(data_dir, 'rb') as file:
        dataset = pickle.load(file)

    return {
        'load' : dataset[train]['load'][load_area],
        'solar': dataset[train]['solar'][renew_area],
        'wind' : dataset[train]['wind'][renew_area],
        'price': dataset[train]['price'][price_area]
    }


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class IEEE123Bus(GridBaseEnv):
    # IEEE 123 bus test feeders
    def _build_net(self):
        self.area = "Distribution Grid"

        # network architecture
        self.net = IEEE123Bus()

        # list of agents
        DG_1 = DG('DG 1', bus='Bus 24',  min_p_mw=0, max_p_mw=0.66, sn_mva=0.825, control_q=True, cost_curve_coefs=[100, 51.6, 0.4615])
        # DG_2 = DG('DG 2', bus='Bus 41',  min_p_mw=0, max_p_mw=0.66, sn_mva=0.825, control_q=True, cost_curve_coefs=[100, 51.6, 0.4615])
        DG_3 = DG('DG 3', bus='Bus 94',  min_p_mw=0, max_p_mw=0.5,  sn_mva=0.625, control_q=True, cost_curve_coefs=[100, 72.4, 0.5011])
        DG_4 = DG('DG 4', bus='Bus 71',  min_p_mw=0, max_p_mw=0.5,  sn_mva=0.625, control_q=True, cost_curve_coefs=[100, 72.4, 0.5011])
        # DG_5 = DG('DG 5', bus='Bus 114', min_p_mw=0, max_p_mw=0.4,  sn_mva=0.5,   control_q=True, cost_curve_coefs=[100, 81.6, 0.3011])
        PV_1 = RES('PV 1', source='SOLAR', bus='Bus 22',  sn_mva=0.1, control_q=False)
        PV_2 = RES('PV 2', source='SOLAR', bus='Bus 250', sn_mva=0.1, control_q=False)
        PV_3 = RES('PV 3', source='SOLAR', bus='Bus 43',  sn_mva=0.1, control_q=False)
        PV_4 = RES('PV 4', source='SOLAR', bus='Bus 450', sn_mva=0.1, control_q=False)
        PV_5 = RES('PV 5', source='SOLAR', bus='Bus 39',  sn_mva=0.1, control_q=False)
        WP_1 = RES('WP_1', source='WIND', bus='Bus 4',  sn_mva=0.1, control_q=False)
        WP_2 = RES('WP_2', source='WIND', bus='Bus 59', sn_mva=0.1, control_q=False)
        WP_3 = RES('WP_3', source='WIND', bus='Bus 46', sn_mva=0.1, control_q=False)
        WP_4 = RES('WP_4', source='WIND', bus='Bus 75', sn_mva=0.1, control_q=False)
        WP_5 = RES('WP_5', source='WIND', bus='Bus 83', sn_mva=0.1, control_q=False)
        TAP_1 = Transformer('TAP 1', type='TAP', fbus='Bus 150', tbus='Bus 149', sn_mva=5., tap_max=2,  tap_min=-2)
        TAP_2 = Transformer('TAP 2', type='TAP', fbus='Bus 9',   tbus='Bus 14',  sn_mva=1., tap_max=16, tap_min=-16)
        TAP_3 = Transformer('TAP 3', type='TAP', fbus='Bus 25',  tbus='Bus 26',  sn_mva=1., tap_max=16, tap_min=-16)
        TAP_4 = Transformer('TAP 4', type='TAP', fbus='Bus 160', tbus='Bus 67',  sn_mva=1., tap_max=2, tap_min=-2)
        SCB_1 = Shunt('SCB 1', bus='Bus 108', q_mvar=-0.3, max_step=4)
        SCB_2 = Shunt('SCB 2', bus='Bus 76',  q_mvar=-0.3, max_step=4)
        ESS_1 = ESS('Storage 1', bus='Bus 20', min_p_mw=-0.5, max_p_mw=0.5, max_e_mwh=2, min_e_mwh=0.2)
        ESS_2 = ESS('Storage 2', bus='Bus 56', min_p_mw=-0.5, max_p_mw=0.5, max_e_mwh=2, min_e_mwh=0.2)
        # ESS_3 = ESS('Storage 3', bus='Bus 113', min_p_mw=-0.25, max_p_mw=0.25, max_e_mwh=1, min_e_mwh=0.1)
        GRID = Grid('GRID', bus='Bus 150', sn_mva=5.)
        SW_1 = Switch('SW 1', fbus='Bus 18', tbus='Bus 135')
        SW_2 = Switch('SW 2', fbus='Bus 13', tbus='Bus 152')
        SW_3 = Switch('SW 3', fbus='Bus 54', tbus='Bus 94')
        SW_4 = Switch('SW 4', fbus='Bus 60', tbus='Bus 160')
        SW_5 = Switch('SW 5', fbus='Bus 97', tbus='Bus 197')

        self.agents = [DG_1, DG_3, DG_4, PV_1, PV_2, PV_3, PV_4, PV_5, WP_1, WP_2, WP_3, WP_4, WP_5, \
            TAP_1, TAP_2, TAP_3, TAP_4, SCB_1, SCB_2, ESS_1, ESS_2, GRID, SW_1, SW_2, SW_3, SW_4, SW_5]

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    @property
    def resource_agents(self):
        return [agent for agent in self.agents if agent.type in ['GRID', 'DG', 'CL', 'ESS', 'SCB', 'SOLAR', 'WIND']]

    @property
    def grid_agent(self):
        return [agent for agent in self.agents if agent.type in ['GRID']]

    @property
    def dg_agents(self):
        return [agent for agent in self.agents if agent.type in ['DG']]

    @property
    def cl_agents(self):
        return [agent for agent in self.agents if agent.type in ['CL']]

    @property
    def res_agents(self):
        return [agent for agent in self.agents if agent.type in ['SOLAR', 'WIND']]

    @property
    def ess_agents(self):
        return [agent for agent in self.agents if agent.type in ['ESS']]

    @property
    def tap_agents(self):
        return [agent for agent in self.agents if agent.type in ['TAP']]

    @property
    def trafo_agents(self):
        return [agent for agent in self.agents if agent.type in ['Trafo']]

    @property
    def shunt_agents(self):
        return [agent for agent in self.agents if agent.type in ['SCB']]

    @property
    def switch_agents(self):
        return [agent for agent in self.agents if agent.type in ['SW']]

    def _reward_and_safety(self, net):
        if net["converged"]:
            # reward and safety
            reward, safety = 0, 0
            for agent in self.agents:
                reward -= agent.cost
                safety += agent.safety
            # update power flow safety
            vm = net.res_bus.vm_pu.values
            loading = net.res_line.loading_percent.values
            overloading = np.maximum(loading - 100, 0).sum()
            overvoltage = np.maximum(vm - 1.05, 0).sum()
            undervoltage = np.maximum(0.95 - vm, 0).sum()
            safety += overloading / 100 + overvoltage + undervoltage
        else:
            reward = -200.0
            safety = 2.0
            vm, loading = np.nan, np.nan
            print('Doesn\'t converge!')

        if self.kwargs.get('penalty_coef'):
            reward -= safety * self.kwargs.get('penalty_coef')
        if self.kwargs.get('safety_scale'):
            safety *= self.kwargs.get('safety_scale')
        # info
        info = {'s': safety}
        info['load'] = net.res_load.p_mw.sum()
        info['loading'] = loading
        info['voltage'] = vm
        
        return reward, info