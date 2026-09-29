from collections import OrderedDict
import numpy as np
from gridages.utils.utils import attach_device_to_net
from gridages.networks.ieee123 import IEEE123Bus
from gridages.devices import (
    DG,
    RES,
    ESS,
    Grid,
    Transformer,
    Shunt,
)

def build_ieee123_devices(task="vvc"):
    """
    Shared IEEE-123 physical scenario.

    task:
        "vvc" -> PV/Wind Q control + regulators + capacitor banks
        "ems" -> same physical system; EMS actions configured later
    """

    if task not in ["vvc", "ems"]:
        raise ValueError(
            "task must be either 'vvc' or 'ems'"
        )

    area = "Distribution Grid"

    # Build IEEE-123 network
   
    net = IEEE123Bus(area)

    #Remove pre-existing DER scenario from the instantiated IEEE 123 network.

    net.sgen.drop(
        net.sgen.index,
        inplace=True
    )

    net.storage.drop(
        net.storage.index,
        inplace=True
    )

    # Distributed Generators

    dg1 = DG(
        name="DG 1",
        bus="Bus 35",
        min_p_mw=0.0,
        max_p_mw=0.60,
        sn_mva=0.75,
        cost_curve_coefs=[
            100,
            51.6,
            0.4615,
        ],
    )

    dg3 = DG(
        name="DG 3",
        bus="Bus 67",
        min_p_mw=0.0,
        max_p_mw=0.60,
        sn_mva=0.75,
        cost_curve_coefs=[
            100,
            72.4,
            0.5011,
        ],
    )

    dg4 = DG(
        name="DG 4",
        bus="Bus 97",
        min_p_mw=0.0,
        max_p_mw=0.50,
        sn_mva=0.625,
        cost_curve_coefs=[
            100,
            72.4,
            0.5011,
        ],
    )

    # Renewable Q configuration
    pv_rating = 0.26175
    wt_rating = 0.305375

    if task == "vvc":
        pv_q_kwargs = {
            "min_q_mvar": -pv_rating,
            "max_q_mvar": pv_rating,
        }

        wt_q_kwargs = {
            "min_q_mvar": -wt_rating,
            "max_q_mvar": wt_rating,
        }
    else:
        pv_q_kwargs = {}
        wt_q_kwargs = {}

    # PV
    
    pv1 = RES(
        name="PV 1",
        bus="Bus 30",
        sn_mva=pv_rating,
        source="solar",
        **pv_q_kwargs,
    )

    pv2 = RES(
        name="PV 2",
        bus="Bus 47",
        sn_mva=pv_rating,
        source="solar",
        **pv_q_kwargs,
    )

    pv3 = RES(
        name="PV 3",
        bus="Bus 66",
        sn_mva=pv_rating,
        source="solar",
        **pv_q_kwargs,
    )

    pv4 = RES(
        name="PV 4",
        bus="Bus 85",
        sn_mva=pv_rating,
        source="solar",
        **pv_q_kwargs,
    )

    pv5 = RES(
        name="PV 5",
        bus="Bus 101",
        sn_mva=pv_rating,
        source="solar",
        **pv_q_kwargs,
    )

    pv6 = RES(
        name="PV 6",
        bus="Bus 114",
        sn_mva=pv_rating,
        source="solar",
        **pv_q_kwargs,
    )
   
    # Wind

    wt1 = RES(
        name="WP 1",
        bus="Bus 18",
        sn_mva=wt_rating,
        source="wind",
        **wt_q_kwargs,
    )

    wt2 = RES(
        name="WP 2",
        bus="Bus 57",
        sn_mva=wt_rating,
        source="wind",
        **wt_q_kwargs,
    )

    wt3 = RES(
        name="WP 3",
        bus="Bus 80",
        sn_mva=wt_rating,
        source="wind",
        **wt_q_kwargs,
    )

    wt4 = RES(
        name="WP 4",
        bus="Bus 109",
        sn_mva=wt_rating,
        source="wind",
        **wt_q_kwargs,
    )

    # wt5 = RES(
    #     name="WP 5",
    #     bus="Bus 83",
    #     sn_mva=0.10,
    #     source="wind",
    #     **wt_q_kwargs,
    # )

    # Energy Storage

    ess1 = ESS(
        name="Storage 1",
        bus="Bus 54",
        min_p_mw=-0.60,
        max_p_mw=0.60,
        capacity=2.4,
        max_e_mwh=2.4,
        min_e_mwh=0.24,
        init_soc=0.50,
    )

    ess2 = ESS(
        name="Storage 2",
        bus="Bus 110",
        min_p_mw=-0.60,
        max_p_mw=0.60,
        capacity=2.4,
        max_e_mwh=2.4,
        min_e_mwh=0.24,
        init_soc=0.50,
    )

    # Grid

    grid = Grid(
        name="GRID",
        bus="Bus 150",
        sn_mva=5.0,
    )

    # Voltage Regulators

    reg1 = Transformer(
        name="Regulator 1",
        sn_mva=5.0,
        tap_min=-2,
        tap_max=2,
    )

    reg2 = Transformer(
        name="Regulator 2",
        sn_mva=1.0,
        tap_min=-16,
        tap_max=16,
    )

    reg3 = Transformer(
        name="Regulator 3",
        sn_mva=1.0,
        tap_min=-16,
        tap_max=16,
    )

    reg4 = Transformer(
        name="Regulator 4",
        sn_mva=1.0,
        tap_min=-2,
        tap_max=2,
    )

    # Switched Capacitor Banks

    scb1 = Shunt(
        name="SCB 1",
        bus="Bus 108",
        q_mvar=-0.30,
        max_step=4,
    )

    scb2 = Shunt(
        name="SCB 2",
        bus="Bus 76",
        q_mvar=-0.30,
        max_step=4,
    )

    # Device Dictionary

    devices = OrderedDict([
        (dg1.name, dg1),
        (dg3.name, dg3),
        (dg4.name, dg4),

        (pv1.name, pv1),
        (pv2.name, pv2),
        (pv3.name, pv3),
        (pv4.name, pv4),
        (pv5.name, pv5),
        (pv6.name, pv6),

        (wt1.name, wt1),
        (wt2.name, wt2),
        (wt3.name, wt3),
        (wt4.name, wt4),
        # (wt5.name, wt5),

        (ess1.name, ess1),
        (ess2.name, ess2),

        (grid.name, grid),

        (reg1.name, reg1),
        (reg2.name, reg2),
        (reg3.name, reg3),
        (reg4.name, reg4),

        (scb1.name, scb1),
        (scb2.name, scb2),
    ])

    #Attach GridAges devices to Pandapower net

    for dev in devices.values():
        attach_device_to_net(
            net,
            area,
            dev
        )

    return net, devices, area