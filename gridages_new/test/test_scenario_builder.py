import pandapower as pp
from gridages.scenario_builder import build_ieee13_scenario

def evaluate_scenario(
        name, 

        pv_bus = None, 
        pv_mw=None,
        pv_penetration=None,
        pv_units=None,

        bess_bus = None,
        bess_power_mw = None,
        bess_penetration=None,
        bess_dispatch_fraction=-1.0,
        bess_duration_h=2.0,
        bess_units = None,

        load_scale=1.0
    ):

    #Build customized IEEE-13 network
    net = build_ieee13_scenario(
        pv_bus = pv_bus,
        pv_mw = pv_mw,
        pv_penetration=pv_penetration,
        pv_units = pv_units,

        bess_bus = bess_bus,
        bess_power_mw = bess_power_mw,
        bess_penetration=bess_penetration,
        bess_dispatch_fraction=bess_dispatch_fraction,
        bess_duration_h = bess_duration_h,
        bess_units=bess_units,

        load_scale=load_scale
    )

    #Run powerflow
    pp.runpp(net, numba = False)

    #Calculate values
    min_voltage = net.res_bus.vm_pu.min()
    max_voltage = net.res_bus.vm_pu.max()
    max_loading = net.res_line.loading_percent.max()
    grid_power = net.res_ext_grid.p_mw.sum()
    total_load = net.load.p_mw.sum()
    total_pv = (
        net.sgen.p_mw.sum()
        if len(net.sgen)>0
        else 0.0
    )
    total_bess = (
        net.storage.p_mw.sum() 
        if len(net.storage) > 0 else 0.0
    )
    total_bess_energy = (
        net.storage["max_e_mwh"].sum()
        if len(net.storage)>0
        else 0.0
    )



    power_loss = grid_power + total_pv - total_load - total_bess

    # Count buses outside acceptable voltage range
    voltage_violations = (
        (net.res_bus.vm_pu < 0.95) |
        (net.res_bus.vm_pu > 1.05)
    ).sum()

    #Find bus with minimum voltage
    min_bus_index = net.res_bus.vm_pu.idxmin()
    min_bus_name = net.bus.loc[min_bus_index, "name"]

    print("\n===================================")
    print(name)
    print("===================================")
    
    print(
        "PV penetration:",
        f"{pv_penetration * 100:.1f}%"
        if pv_penetration is not None
        else "N/A"
    )

    print(
        "Total installed PV:",
        round(total_pv, 6),
        "MW"
    )

    print(
        "BESS penetration:",
        f"{bess_penetration * 100:.1f}%"
        if bess_penetration is not None
        else "N/A"
    )

    print(
        "BESS operating power:",
        round(total_bess, 6),
        "MW"
    )

    print(
        "BESS energy capacity:",
        round(total_bess_energy, 6),
        "MWh"
    )

    print("Minimum voltage:",
          round(min_voltage, 5),
          "pu")

    print("Minimum-voltage bus:",
          min_bus_name)

    print("Maximum voltage:",
          round(max_voltage, 5),
          "pu")
    
    print("Voltage violations:",
      voltage_violations)

    print("Maximum line loading:",
          round(max_loading, 2),
          "%")

    print("Grid active power:",
          round(grid_power, 4),
          "MW")
    
    print("Active power loss:",
      round(power_loss, 4),
      "MW")

    return net

if __name__ == "__main__":

    #S0 - No PV

    evaluate_scenario(
        name="S0 - Base case"
    )

    #S1-N: PV near feeder head
    evaluate_scenario(
        name ="S1-N - 25% PV AT BUS 632",
        pv_bus = 632,
        pv_penetration = 0.25
    )

    #S1-M: PV in the middle of the feeder
    evaluate_scenario(
        name = "S1-M - 25% PV at BUS 671",
        pv_bus=671,
        pv_penetration=0.25
    )

    #S1-F: PV at remote bus
    evaluate_scenario(
        name="S1-F - 25% PV AT BUS 675",
        pv_bus=675,
        pv_penetration=0.25
    )

    #S2: PV + BESS at remote bus
    #bess_power_mw=-0.5 means the battery is discharging 0.5 MW into the network
    #In pandapower's storage sign convention, positive p_mw is charging and negative is discharging.
    evaluate_scenario(
        name="S2 - 25% PV 675 + 12.5% BESS 675",
        pv_bus = 675,
        pv_penetration=0.25,
        bess_bus = 675,
        bess_penetration=0.125,
        bess_dispatch_fraction=-1.0,
        bess_duration_h=2.0
    )

    #S3: PV remote, BESS more centrally located
    evaluate_scenario(
        name = "S3 - 25% PV 675 + 12.5% BESS 671",
        pv_bus=675,
        pv_penetration=0.25,
        bess_bus=671,
        bess_penetration=0.125,
        bess_dispatch_fraction=-1.0,
        bess_duration_h=2.0
    )

    #S4: Distributed PV
    evaluate_scenario(
        name="S4 - 25% DISTRIBUTED PV",
        pv_penetration=0.25,
        pv_units=[
            {"bus": 646, "share": 1/3},
            {"bus": 652, "share": 1/3},
            {"bus":675, "share":1/3}
        ]
    )

    #S5: Distributed PV + Distributed BESS
    evaluate_scenario(
        name="S5 - 25% DISTRIBUTED PV + 12.5% DISTRIBUTED BESS",
        pv_penetration=0.25,
        pv_units=[
            {"bus": 646, "share":1/3},
            {"bus":652, "share": 1/3},
            {"bus": 675, "share": 1/3}
        ],

        bess_penetration=0.125,
        bess_dispatch_fraction=-1.0,
        bess_duration_h=2.0,

        bess_units = [
            {"bus":646, "share": 1/3},
            {"bus": 652, "share": 1/3},
            {"bus":675, "share":1/3}
        ]
    )




# def test_ieee13_base_scenario():

#     #Build original IEEE-13 network

#     net = build_ieee13_scenario()

#     #run power flow

#     pp.runpp(net, numba=False)

#     #Print basic information

#     print("\n===========================")
#     print("IEEE-13 BASE CASE")
#     print("===========================")

#     print("Number of buses:", len(net.bus))

#     print("Total active load:",
#           net.load.p_mw.sum(), "MW")
    
#     print("Total reactive load:",
#           net.load.q_mvar.sum(), "MVar")
    
#     print("Minimum voltage:",
#           net.res_bus.vm_pu.min(), "pu")
    
#     print("Maximum voltage:",
#           net.res_bus.vm_pu.max(), "pu")
    
#     print("Maximum line loading:",
#           net.res_line.loading_percent.max(), "%")
    
#     print("Grid active power:",
#           net.res_ext_grid.p_mw.sum(), "MW")
    
#     print("\nBus Voltages")
#     print("-------------------------")

#     for i in net.bus.index:

#         bus_name = net.bus.loc[i, "name"]
#         voltage = net.res_bus.loc[i, "vm_pu"]

#         print(bus_name, ":", round(voltage, 5), "pu")
    
#     assert net.converged

#     assert len(net.bus) >0
#     assert net.res_bus.vm_pu.notna().all()

# if __name__ == "__main__":
#     test_ieee13_base_scenario()
    
