from gridages.envs.single_agent.scenarios.ieee13_env import (
    IEEE13ScenarioEnv
)

def test_environment():

    env = IEEE13ScenarioEnv(
        env_config={
            "episode_length": 24,

            "train": False,

            "store_info": [
                "operating_cost",
                 "bus_voltage",
                "line_loading",
             ],
                
                "store_summaries": True,
        }
    )

    print(
        "\n==============================="
    )
    print(
        "IEEE-13 GRIDAGES ENVIRONMENT"
    )
    print(
        "================================"
    )
    print(
        "\nAction space:"
    )
    print(
        env.action_space
    )

    print(
        "\nObservation space:"
    )
    print(
        "\nRegistered devices:"
    )

    for name, dev, in env.devices.items():

        print(
            name,
            "|",
            dev.__class__.__name__,
            "|",
            getattr(dev, "bus", None),
            "| action dim:",
            dev.action.dim_c
        )

    #Reset

    obs, info = env.reset(
        seed = 0
    )

    print(
        "\nObservation shape:",
        obs.shape
    )

    #One random action
    action = (
        env.action_space.sample()
    )

    print(
        "\nSample action:"
    )
    print(
        action
    )

    #One environment step

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    print(
        "\n Power flow converged:",
        info.get("converged")
    )

    print(
        "Reward:",
        reward
    )

    print(
        "Safety:",
        info.get("s")
    )
    print(
        "Minimum voltage:",
        info.get(
            "bus_voltage_min"
        )
    )
    print(
        "Maximum line loading:",
        info.get(
            "line_loading_max"
        )
    )
    print(
        "Terminated:",
        terminated
    )

if __name__ == "__main__":
    test_environment()