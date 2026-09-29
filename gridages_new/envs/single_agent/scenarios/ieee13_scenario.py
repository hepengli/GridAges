from collections import OrderedDict
from gridages.networks.ieee13 import IEEE13Bus
from gridages.devices import RES, ESS, DG, Grid, Transformer
import numpy as np

def build_ieee13_devices(task="ems"):
    if task not in {"ems", "vvc"}:
        raise ValueError(
            "task must be either 'ems' or 'vvc'"
        )

    area = "MG1"

    #Base IEEE-13 feeder
    net = IEEE13Bus(area)

    #Native feeder load
    base_load_mw = net.load["p_mw"].sum()

    #Normalized device sizing

    pv_penetration = 0.15
    wind_penetration = 0.1
    bess_penetration = 0.125

    pv_rating_mw = pv_penetration * base_load_mw

    wind_rating_mw = (
        wind_penetration * base_load_mw
    )

    bess_power_mw = (
        bess_penetration * base_load_mw
    )

    bess_duration_h = 2.0

    bess_energy_mwh = (
        bess_power_mw * bess_duration_h
    )

    dg_max_mw =(
        0.1*base_load_mw
    )

    #Renewable Devices
    if task == "vvc":
        pv1 = RES(
            name="PV1",
            bus="Bus 675",
            sn_mva=pv_rating_mw,
            source="solar",

            #VVC controls reactive power
            min_q_mvar=-pv_rating_mw,
            max_q_mvar=pv_rating_mw,

        )
    else:

        pv1 = RES(
            name="PV1",
            bus="Bus 675",
            sn_mva=pv_rating_mw,
            source="solar",
        )

    if task == "vvc":

        wind1 = RES(
            name="WT1",
            bus="Bus 645",
            sn_mva=wind_rating_mw,
            source="wind",

            min_q_mvar=-wind_rating_mw,
            max_q_mvar=wind_rating_mw,
        )
    else:
        wind1 = RES(
            name="WT1",
            bus="Bus 645",
            sn_mva=wind_rating_mw,
            source="wind",
        )

    #Battery

    ess1 = ESS(
        name ="ESS1",
        bus="Bus 671",

        min_p_mw=-bess_power_mw,
        max_p_mw=bess_power_mw,

        capacity=bess_energy_mwh,

        max_e_mwh=0.9*bess_energy_mwh,
        min_e_mwh=0.1*bess_energy_mwh,

        init_soc=0.5
    )

    #Optional DG
    dg1 = DG(
        name="DG1",
        bus="Bus 611",

        min_p_mw=0.0,
        max_p_mw = 0.1 * base_load_mw,

        cost_curve_coefs=[
            100,
            60,
            0.5
        ]
    )

    #Utility Grid

    grid = Grid(
        name="Grid",
        bus="Bus 650",
        sn_mva=5.0,
        sell_discount=0.9
    )

    #Existing transformers
    
    trafos = []

    for _, row in net.trafo.iterrows():

        name = row["name"][len(area)+1:]

        tap_min = row.get(
            "tap_min",
            np.nan
        )

        tap_max = row.get(
            "tap_max",
            np.nan
        )

        #for VVC, expose tap action
        #only when the physical transformer
        #actually has tap limits

        if(
            task == "vvc"
            and not np.isnan(tap_min)
            and not np.isnan(tap_max)
        ):

            trafo = Transformer(
                name=name,
                sn_mva=row["sn_mva"],

                tap_min=int(tap_min),
                tap_max=int(tap_max),
            )

        else:
            trafo= Transformer(
                name=name,
                sn_mva=row["sn_mva"],
            )
        trafos.append(
            (
                    (name, trafo)
            )
                
        )
        

    #Register devices

    devices = OrderedDict([
        (ess1.name, ess1),
        (dg1.name, dg1),
        (pv1.name, pv1),
        (wind1.name, wind1),
        (grid.name, grid),
        *trafos
    ])

    return net, devices, area
