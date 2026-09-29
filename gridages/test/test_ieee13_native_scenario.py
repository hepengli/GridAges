from gridages.envs.single_agent.scenarios.ieee13_scenario import (
    build_ieee13_devices
)


def test_scenario():

    net, devices, area = build_ieee13_devices()

    print("\n===================================")
    print("IEEE-13 NATIVE DEVICE SCENARIO")
    print("===================================")

    print(
        "Base active load:",
        round(net.load.p_mw.sum(), 6),
        "MW"
    )

    print("\nRegistered GridAges devices:")

    for name, dev in devices.items():

        print(
            name,
            "| class:",
            dev.__class__.__name__,
            "| bus:",
            getattr(dev, "bus", None)
        )


if __name__ == "__main__":
    test_scenario()