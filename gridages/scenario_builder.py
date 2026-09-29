import pandapower as pp
from gridages.networks.ieee13 import IEEE13Bus


def get_bus_index(net, bus_number):
    #Convert IEEE bus number such as 632 or 675 into the pandapower bus index.

    bus_name = f"Bus {bus_number}"
    matches = net.bus.index[
        net.bus["name"] == bus_name
    ].tolist()

    if len(matches) == 0:
        raise ValueError(f"{bus_name} not found in the network.")
    
    return matches[0]

def add_pv(net, bus_number, p_mw, q_mvar=0.0):
    #Add a PV generator to selected bus

    bus = get_bus_index(net, bus_number)

    #create_sgen = create static generatior in pandapower. 
    # It represents a generator whose active and reactive power are specified directly. p = p_mw, q = p_mvar
    pp.create_sgen(
        net,
        bus=bus,
        p_mw=p_mw,
        q_mvar=q_mvar,
        name=f"PV at Bus {bus_number}",
        type="PV"
    )

    return net

def add_bess(
        net,
        bus_number,
        p_mw=0.0,
        energy_mwh = 1.0,
        soc_percent = 50.0
):
    
    #Add a battery energy storage system

    bus = get_bus_index(net, bus_number)

    pp.create_storage(
        net,
        bus=bus,
        p_mw = p_mw,
        max_e_mwh = energy_mwh,
        soc_percent = soc_percent,
        name = f"BESS at Bus {bus_number}"
    )

    return net

def build_ieee13_scenario(
        pv_bus=None,
        pv_mw=None,
        pv_penetration=None,
        pv_units=None,

        bess_bus = None,
        bess_power_mw = None,
        bess_energy_mwh=None,
        bess_penetration = None,
        bess_dispatch_fraction=-1.0,
        bess_duration_h=2.0,
        bess_units = None,

        load_scale=1.0
):
    #Build a customized IEEE-13 scenario
    net = IEEE13Bus()

    base_load_mw = net.load["p_mw"].sum()

    #Modify loading condition

    net.load["p_mw"] *= load_scale
    net.load["q_mvar"] *= load_scale

    #Add PV if requested

    if pv_units is not None:

        #Normalized penetration-based distributed PV
        if pv_penetration is not None:
            total_pv_mw = pv_penetration * base_load_mw
            total_share = sum(pv["share"] for pv in pv_units)

            if abs(total_share - 1.0)> 1e-6:
                raise ValueError("PV shares must sum to 1.0")
            
            for pv in pv_units:
                unit_pv_mw = total_pv_mw * pv["share"]

                add_pv(
                    net,
                    bus_number=pv["bus"],
                    p_mw = unit_pv_mw
                )
            #Old absolute_MW method    
        else:
        
            for pv in pv_units:

                add_pv(
                    net,
                    bus_number = pv["bus"],
                    p_mw = pv["p_mw"]
                )
    #Single PV                
    elif pv_bus is not None:
        if pv_penetration is not None:
            pv_power = pv_penetration * base_load_mw
        elif pv_mw is not None:
            pv_power = pv_mw
        else:
            raise ValueError(
                    "PV bus specified but no PV size was provided."
                )

        add_pv(
            net,
            bus_number=pv_bus,
            p_mw=pv_power
        )
        
    #Add battery if requested
    if bess_units is not None:

        #penetration-based distributed BESS
        if bess_penetration is not None:
            total_bess_rating_mw = (
                bess_penetration * base_load_mw
            )
            total_share=sum(
                bess["share"] for bess in bess_units
            )

            if abs(total_share-1.0)>1e-6:
                raise ValueError("BESS shares must sum to 1.0")
            
            for bess in bess_units:
                unit_rating_mw = (
                    total_bess_rating_mw * bess["share"]
                )
                unit_power_mw = (
                    bess_dispatch_fraction * unit_rating_mw
                )
                unit_energy_mwh =(
                    unit_rating_mw * bess_duration_h
                )
                add_bess(
                    net,
                    bus_number=bess["bus"],
                    p_mw=unit_power_mw,
                    energy_mwh=unit_energy_mwh
                )
        else:

            for bess in bess_units:

                add_bess(
                net,
                bus_number = bess["bus"],
                p_mw = bess["p_mw"],
                energy_mwh = bess["energy_mwh"]
            )
    elif bess_bus is not None:
        if bess_penetration is not None:
            bess_rating_mw =(
                bess_penetration*base_load_mw
            )
            bess_power = (
                bess_dispatch_fraction*bess_rating_mw
            )
            bess_energy = (
                bess_rating_mw*bess_duration_h
            )
        elif bess_power_mw is not None:
            bess_power=bess_power_mw

            if bess_energy_mwh is not None:
                bess_energy=bess_energy_mwh
            else:
                bess_energy = (
                    abs(bess_power_mw)*bess_duration_h
                )

        else:
            raise ValueError(
                "BESS bus specified but no BESS size was provided."
            )

        add_bess(
            net,
            bus_number = bess_bus,
            p_mw = bess_power,
            energy_mwh = bess_energy
        )

    return net


