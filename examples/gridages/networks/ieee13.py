import math
import pandapower as pp

from gridages.networks.lines import *


def IEEE13Bus(name=''):
    """Create a balanced positive-sequence approximation of the IEEE 13-node feeder.

    The source data are based on the IEEE PES 13-node distribution test feeder.
    The original feeder is strongly unbalanced and contains single-, two-, and
    three-phase branches and loads.  GridAges uses pandapower's balanced ``runpp``
    model here, so phase quantities are aggregated while preserving:

      * feeder topology and line lengths,
      * positive-sequence line impedances,
      * aggregate P/Q at each load location,
      * ZIP load model type (PQ / I / Z),
      * the 632-671 distributed load via the standard Bus 670 equivalent,
      * the 675 and 611 capacitor banks,
      * the 633-634 transformer,
      * the 671-692 normally closed switch, and
      * a fixed balanced equivalent of the feeder-head voltage regulator.

    Notes
    -----
    The published IEEE benchmark power-flow results start at Bus 650 at 1.0 pu.
    Therefore Bus 650 is retained as the pandapower external-grid bus rather than
    explicitly modeling the upstream 115/4.16-kV substation transformer.  This is
    also compatible with GridAges environments that attach the Grid device to
    ``Bus 650``.

    The physical IEEE regulator is a three-single-phase bank with different phase
    taps.  Since this network is balanced, it is represented by one fixed-ratio,
    low-impedance three-phase equivalent whose ratio is the mean of the published
    phase ratios (1.0625, 1.0500, 1.06875).

    Parameters
    ----------
    name : str, optional
        Prefix used by GridAges when several feeders are combined.

    Returns
    -------
    pandapowerNet
        Balanced pandapower representation of the IEEE 13-node feeder.
    """
    net = pp.create_empty_network(name)
    prefix = f"{name} " if len(name) > 0 else ""

    # ------------------------------------------------------------------
    # Line data
    # ------------------------------------------------------------------
    # The CF-60x types in gridages.networks.lines are positive-sequence
    # equivalents derived from the IEEE phase impedance matrices.
    pp.create_std_type(net, CF601, name='CF-601', element='line')
    pp.create_std_type(net, CF602, name='CF-602', element='line')
    pp.create_std_type(net, CF603, name='CF-603', element='line')
    pp.create_std_type(net, CF604, name='CF-604', element='line')
    pp.create_std_type(net, CF605, name='CF-605', element='line')
    pp.create_std_type(net, CF606, name='CF-606', element='line')
    pp.create_std_type(net, CF607, name='CF-607', element='line')

    # Exact ft -> km conversion used by the IEEE feeder data.
    ft_to_km = 0.0003048

    # ------------------------------------------------------------------
    # Buses
    # ------------------------------------------------------------------
    # Bus 650 is considered a 4.16-kV feeder bus in the IEEE test case.
    bus_650 = pp.create_bus(net, name=f'{prefix}Bus 650', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)

    # RG60 is the regulator output node used in the published IEEE solution.
    bus_rg60 = pp.create_bus(net, name=f'{prefix}RG60', vn_kv=4.16,
                             type='b', zone='13_BUS', max_vm_pu=1.10, min_vm_pu=0.95)

    bus_632 = pp.create_bus(net, name=f'{prefix}Bus 632', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_633 = pp.create_bus(net, name=f'{prefix}Bus 633', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_634 = pp.create_bus(net, name=f'{prefix}Bus 634', vn_kv=0.48,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_645 = pp.create_bus(net, name=f'{prefix}Bus 645', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_646 = pp.create_bus(net, name=f'{prefix}Bus 646', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_671 = pp.create_bus(net, name=f'{prefix}Bus 671', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_680 = pp.create_bus(net, name=f'{prefix}Bus 680', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_684 = pp.create_bus(net, name=f'{prefix}Bus 684', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_611 = pp.create_bus(net, name=f'{prefix}Bus 611', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_652 = pp.create_bus(net, name=f'{prefix}Bus 652', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_692 = pp.create_bus(net, name=f'{prefix}Bus 692', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_675 = pp.create_bus(net, name=f'{prefix}Bus 675', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)

    # Bus 670 is the standard concentrated representation of the load that is
    # distributed along line 632-671, located 1/3 of the way from Bus 632.
    bus_670 = pp.create_bus(net, name=f'{prefix}Bus 670', vn_kv=4.16,
                            type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)


    # ------------------------------------------------------------------
    # Source and feeder-head regulator
    # ------------------------------------------------------------------
    # The published IEEE solution treats Bus 650 as 1.0 pu.  Keeping the
    # external grid here also preserves compatibility with GridAges' Grid
    # device, which is attached to Bus 650 in ieee13_mg.py.
    pp.create_ext_grid(net, bus_650, vm_pu=1.0, name=f'{prefix}Grid')

    # OpenDSS represents each regulator unit as a very low-impedance
    # transformer (XHL=0.01%, load loss=0.01%).  For the balanced equivalent,
    # vkr_percent is R% and vk_percent is |R+jX|%.
    regulator_r_percent = 0.01
    regulator_x_percent = 0.01

    # A controllable balanced equivalent of the feeder-head voltage regulator.
    # Since this network is balanced, it is represented by one
    # three-phase equivalent ratio tap changer with 0.625% tap steps.
    # The physical IEEE regulator is a three-single-phase bank with different
    # phase taps:
    # A = 1.0625, B = 1.0500, C = 1.06875.
    # Therefore, IEEE phase tap positions are approximately A=10, B=8, C=11.
    # The balanced model uses tap 10 as the nearest integer to their mean.
    pp.create_transformer_from_parameters(
        net,
        hv_bus=bus_650,
        lv_bus=bus_rg60,
        sn_mva=5.0,
        vn_hv_kv=4.16,
        vn_lv_kv=4.16,
        vkr_percent=regulator_r_percent,
        vk_percent=math.hypot(
            regulator_r_percent,
            regulator_x_percent,
        ),
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="lv",
        tap_neutral=0,
        tap_min=-16,
        tap_max=16,
        tap_step_percent=0.625,
        tap_pos=10,
        tap_changer_type="Ratio",
        name=f"{prefix}Regulator 1",
    )

    # ------------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------------
    # 650/RG60 -> 632: 2000 ft, configuration 601.
    pp.create_line(net, bus_rg60, bus_632, length_km=2000 * ft_to_km,
                   std_type='CF-601', name=f'{prefix}Line 650-632')

    pp.create_line(net, bus_632, bus_645, length_km=500 * ft_to_km,
                   std_type='CF-603', name=f'{prefix}Line 632-645')
    pp.create_line(net, bus_632, bus_633, length_km=500 * ft_to_km,
                   std_type='CF-602', name=f'{prefix}Line 632-633')
    pp.create_line(net, bus_645, bus_646, length_km=300 * ft_to_km,
                   std_type='CF-603', name=f'{prefix}Line 645-646')
    pp.create_line(net, bus_684, bus_652, length_km=800 * ft_to_km,
                   std_type='CF-607', name=f'{prefix}Line 684-652')

    # The IEEE distributed load on 632-671 is represented at Bus 670, 1/3 of
    # the way from Bus 632.  Preserve the original total length of 2000 ft.
    pp.create_line(net, bus_632, bus_670, length_km=667 * ft_to_km,
                   std_type='CF-601', name=f'{prefix}Line 632-670')
    pp.create_line(net, bus_670, bus_671, length_km=1333 * ft_to_km,
                   std_type='CF-601', name=f'{prefix}Line 670-671')

    pp.create_line(net, bus_671, bus_684, length_km=300 * ft_to_km,
                   std_type='CF-604', name=f'{prefix}Line 671-684')
    pp.create_line(net, bus_671, bus_680, length_km=1000 * ft_to_km,
                   std_type='CF-601', name=f'{prefix}Line 671-680')
    pp.create_line(net, bus_684, bus_611, length_km=300 * ft_to_km,
                   std_type='CF-605', name=f'{prefix}Line 684-611')
    pp.create_line(net, bus_692, bus_675, length_km=500 * ft_to_km,
                   std_type='CF-606', name=f'{prefix}Line 692-675')

    # Normally closed sectionalizing switch between 671 and 692.
    pp.create_switch(
        net,
        bus=bus_671,
        element=bus_692,
        et='b',
        type='LS',
        closed=True,
        z_ohm=1e-4,
        name=f'{prefix}Switch 1',
    )

    # ------------------------------------------------------------------
    # 633-634 distribution transformer
    # ------------------------------------------------------------------
    # IEEE data: 500 kVA, 4.16/0.48 kV, R=1.1%, X=2.0%.
    # pandapower expects vkr_percent=R and vk_percent=|R+jX|.
    xfm_r_percent = 1.1
    xfm_x_percent = 2.0

    pp.create_transformer_from_parameters(
        net,
        hv_bus=bus_633,
        lv_bus=bus_634,
        sn_mva=0.5,
        vn_hv_kv=4.16,
        vn_lv_kv=0.48,
        vkr_percent=xfm_r_percent,
        vk_percent=math.hypot(xfm_r_percent, xfm_x_percent),
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        name=f'{prefix}Transformer 1',
    )

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------
    # Three-phase / phase-specific powers from the IEEE case are summed into
    # one balanced load per location.  The original load MODEL is retained:
    #   PQ -> default constant-power load
    #   I  -> const_i_percent=100
    #   Z  -> const_z_percent=100

    # Spot loads
    pp.create_load(net, bus_634, p_mw=0.400, q_mvar=0.290,
                   name=f'{prefix}Load 634 spot')                       # Y-PQ
    pp.create_load(net, bus_645, p_mw=0.170, q_mvar=0.125,
                   name=f'{prefix}Load 645 spot')                       # Y-PQ
    pp.create_load(net, bus_646, p_mw=0.230, q_mvar=0.132,
                   const_z_percent=100.0,
                   name=f'{prefix}Load 646 spot')                       # D-Z
    pp.create_load(net, bus_652, p_mw=0.128, q_mvar=0.086,
                   const_z_percent=100.0,
                   name=f'{prefix}Load 652 spot')                       # Y-Z
    pp.create_load(net, bus_671, p_mw=1.155, q_mvar=0.660,
                   name=f'{prefix}Load 671 spot')                       # D-PQ
    pp.create_load(net, bus_675, p_mw=0.843, q_mvar=0.462,
                   name=f'{prefix}Load 675 spot')                       # Y-PQ
    pp.create_load(net, bus_692, p_mw=0.170, q_mvar=0.151,
                   const_i_percent=100.0,
                   name=f'{prefix}Load 692 spot')                       # D-I
    pp.create_load(net, bus_611, p_mw=0.170, q_mvar=0.080,
                   const_i_percent=100.0,
                   name=f'{prefix}Load 611 spot')                       # Y-I

    # Distributed load on line 632-671, concentrated at Bus 670 as in the
    # standard OpenDSS implementation: 17+66+117 = 200 kW,
    # 10+38+68 = 116 kVAr.
    pp.create_load(net, bus_670, p_mw=0.200, q_mvar=0.116,
                   name=f'{prefix}Load 632-671 distributed')            # Y-PQ

    # ------------------------------------------------------------------
    # Capacitor banks
    # ------------------------------------------------------------------
    # pandapower uses the load convention for shunt results, so capacitive
    # injection is represented by negative q_mvar.
    pp.create_shunt(
        net,
        bus=bus_675,
        q_mvar=-0.600,
        p_mw=0.0,
        vn_kv=4.16,
        name=f'{prefix}Capacitor 675',
    )
    pp.create_shunt(
        net,
        bus=bus_611,
        q_mvar=-0.100,
        p_mw=0.0,
        vn_kv=4.16,
        name=f'{prefix}Capacitor 611',
    )

    return net


if __name__ == '__main__':
    net = IEEE13Bus()
    pp.runpp(net, voltage_depend_loads=True)

    print(net.res_bus)
    print(net.res_ext_grid)
    print(net.trafo)
    print('Total nominal load P [MW]:', net.load['p_mw'].sum())
    print('Total nominal load Q [MVAr]:', net.load['q_mvar'].sum())
    print('Total capacitor Q [MVAr]:', net.shunt['q_mvar'].sum())
