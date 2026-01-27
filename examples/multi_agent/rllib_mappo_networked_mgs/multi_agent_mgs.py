from gridages.envs.multi_agent.ieee34_ieee13 import MultiAgentMicrogrids

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.tune.registry import get_trainable_cls, register_env

from ray.tune.logger import UnifiedLogger

import os
LOG_DIR = os.path.abspath("./ray_logs")

def logger_creator(config):
    os.makedirs(LOG_DIR, exist_ok=True)
    return UnifiedLogger(config, LOG_DIR, loggers=None)


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)

parser.set_defaults(
    checkpoint_freq=1000,
    verbose=0,
    no_tune=True,
    number_agents=3,
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Here, we use the "Agent Environment Cycle" (AEC) PettingZoo environment type.
    # For a "Parallel" environment example, see the rock paper scissors examples
    # in this same repository folder.
    env_config = {
    "train": True,
    "penalty": 10,
    "share_reward": False,
    }
    register_env("env", lambda _: ParallelPettingZooEnv(
            MultiAgentMicrogrids(env_config)
        )
    )

    # Create an env instance ONCE to extract spaces for each agent.
    probe_env = MultiAgentMicrogrids(env_config)
    agent_ids = list(probe_env.possible_agents)

    obs_spaces = {aid: probe_env.observation_spaces[aid] for aid in agent_ids}
    act_spaces = {aid: probe_env.action_spaces[aid] for aid in agent_ids}

    # Policies are called just like the agents (exact 1:1 mapping).
    policies = set(agent_ids)
    rl_module_specs = {
        "MG1": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_spaces["MG1"],
            action_space=act_spaces["MG1"],
            model_config=DefaultModelConfig(fcnet_hiddens=[128, 128]),
            catalog_class=PPOCatalog,
        ),
        "MG2": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_spaces["MG2"],
            action_space=act_spaces["MG2"],
            model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]),
            catalog_class=PPOCatalog,
        ),
        "MG3": RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_spaces["MG3"],
            action_space=act_spaces["MG3"],
            model_config=DefaultModelConfig(fcnet_hiddens=[64, 128]),
            catalog_class=PPOCatalog,
        ),
    }

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("env")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda aid, *a, **k: aid),
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=rl_module_specs,
            ),
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(num_env_runners=0)
    )

    run_rllib_example_script_experiment(base_config, args)