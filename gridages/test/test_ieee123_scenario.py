from gridages.envs.single_agent.scenarios.ieee123_scenario import (
    build_ieee123_devices,
)

def main():

    net, devices, area = build_ieee123_devices(
        task="vvc"
    )

    print("\n===============================")
    print("IEEE-123 SCENARIO")
    print("===============================")

    print("\nArea:")
    print(area)

    print("\nNetwork:")
    print("Buses:", len(net.bus))
    print("Lines:", len(net.line))
    print("Loads:", len(net.load))
    print("Transformers:", len(net.trafo))
    print("Static generators:", len(net.sgen))
    print("Storage:", len(net.storage))
    print("Shunts:", len(net.shunt))

    print(
        "\nTotal P load:",
        net.load["p_mw"].sum(),
        "MW"
    )

    print(
        "Total Q load:",
        net.load["q_mvar"].sum(),
        "MVAr"
    )

    print("\nRegistered GridAges devices:")

    for name, dev in devices.items():

        print(
            name,
            "|",
            dev.__class__.__name__,
            "| Bus:",
            getattr(dev, "bus", None),
            "| continuous:",
            dev.action.dim_c,
            "| discrete:",
            dev.action.dim_d,
        )

if __name__ == "__main__":
    main()