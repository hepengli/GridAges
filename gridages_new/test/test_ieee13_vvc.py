from gridages.envs.single_agent.scenarios.ieee13_vvc import (
    IEEE13VVCEnv
)

def test_vvc():

    env = IEEE13VVCEnv(
        env_config={
            "episode_length": 24,
            "train": False,

            "store_info": [
                "bus_voltage",
                "line_loading",
            ],

            "store_summaries": True,

            "omega_loss": 100.0,
            "omega_q": 1.0,
            "omega_v": 1.0,
        }
    )

    print(
        "\n=========================="
    )

    print(
        "IEEE-13 VVC environment"
    )

    print(
        "============================="
    )

    print(
        "\nAction space:"
    )
    print(
        env.action_space
    )
    print(
        "\nRegistered devices:"
    )

    for (
        name,
        dev
    ) in env.devices.items():
        
        print(
            name,
            "|",
            dev.__class__.__name__,
            "| Continuous dim:",
            dev.action.dim_c,
            "| discrete dim:",
            dev.action.dim_d,
        )

    obs, info = env.reset(
        seed = 0
    )

    print(
        "\n Observation shape:",
        obs.shape
    )

    action =(
        env.action_space.sample()
    )

    print(
        "\nSample VVC action:"
    )

    print(
        action
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info
    )= env.step(
        action
    )

    print(
        "\nPower flow converged:",
        info.get(
            "converged"
        )
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
        "\nVVC components:"
    )

    print(
        env.last_vvc_terms
    )

    print(
        "\nMinimum voltage:",
        info.get(
            "bus_voltage_min"
        )
    )

    print(
        "Maximum voltage:",
        info.get(
            "bus_voltage_max"
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
    test_vvc()