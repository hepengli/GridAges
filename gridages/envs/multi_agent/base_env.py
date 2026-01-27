from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandapower as pp

class GridAgesParallelEnv(ParallelEnv):
    metadata = {"name": "grid_ages_parallel_v0"}

    def __init__(self, config):
        self.config = config
        self.max_episode_steps = config.get("episode_length", 24)

        # build pandapower net + dataset + agent controllers
        self.net = self._build_network()
        self.dataset = self._load_dataset()

        self._t0 = 0
        self._t = 0
        self._step_count = 0

        # controllers: dict[str, MicrogridAgent]
        self.controllers = self._build_controllers()

        self.possible_agents = list(self.controllers.keys())
        self.agents = self.possible_agents[:]  # alive agents

        # cache spaces
        self._action_spaces = {
            aid: self.controllers[aid].action_space()
            for aid in self.possible_agents
        }
        self._observation_spaces = {
            aid: self.controllers[aid].observation_space()
            for aid in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0
        self.agents = self.possible_agents[:]

        # choose start time
        self._t0 = self._sample_start_index(options)
        self._t = self._t0

        # reset net + controllers
        self.net = self._build_network()
        for ctrl in self.controllers.values():
            ctrl.reset(self.net, t=self._t)

        # initial PF (optional but often useful)
        converged = self._run_powerflow()

        obs = {aid: self.controllers[aid].observe(self.net, t=self._t) for aid in self.agents}
        infos = {aid: {"converged": converged} for aid in self.agents}
        return obs, infos

    def step(self, actions):
        # actions: dict[agent_id, action]
        if not self.agents:
            raise RuntimeError("step() called after all agents are done")

        # 1) apply exogenous profiles for this timestep (loads, solar, wind, etc.)
        self._apply_profiles(t=self._t)

        # 2) apply each agent's actions to its devices / net
        for aid, act in actions.items():
            self.controllers[aid].apply_action(self.net, act, t=self._t)

        # 3) run PF once
        converged = self._run_powerflow()

        # 4) compute obs/reward/termination
        obs = {aid: self.controllers[aid].observe(self.net, t=self._t) for aid in self.agents}

        rewards = {aid: self.controllers[aid].reward(self.net, t=self._t, converged=converged)
                   for aid in self.agents}

        # global termination conditions (episode length, non-convergence, etc.)
        self._step_count += 1
        self._t += 1

        terminations = {aid: False for aid in self.agents}
        truncations = {aid: (self._step_count >= self.max_episode_steps) for aid in self.agents}

        # if you want "hard fail" termination on non-convergence:
        # if not converged: terminations = {aid: True for aid in self.agents}

        infos = {aid: {"converged": converged} for aid in self.agents}

        # If the episode ended, PettingZoo expects env.agents to become []
        if all(truncations.values()) or all(terminations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _run_powerflow(self):
        try:
            pp.runpp(self.net)
            self.net["converged"] = True
            return True
        except Exception:
            self.net["converged"] = False
            return False
