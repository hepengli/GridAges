from typing import Optional
import numpy as np

from .storage import ESS


class EV(ESS):
    """Electric vehicle modeled as a mobile ESS.

    The EV is connected only during its parking interval.

    `time` supplied by the environment may be a global dataset index.  EV
    connection schedules are daily, so connection logic uses time modulo
    `daily_steps`.

    Parameters
    ----------
    external_control : bool
        If True, the EV uses the action supplied by the controller/agent.
        If False, the EV automatically charges as fast as possible after
        arrival until it reaches max_soc.
    """

    def __init__(
        self,
        name: str,
        bus: str,
        min_p_mw: float,
        max_p_mw: float,
        capacity: float,
        *,
        max_e_mwh: Optional[float] = None,
        min_e_mwh: float = 0.0,
        init_soc: float = 0.5,
        min_q_mvar: float = np.nan,
        max_q_mvar: float = np.nan,
        sn_mva: float = np.nan,
        ch_eff: float = 0.98,
        dsc_eff: float = 0.98,
        cost_curve_coefs=(0.0, 0.0, 0.0),
        dt: float = 1.0,
        arrive_time: int = 0,
        depart_time: int = 0,
        external_control: bool = True,
        usage_type: Optional[str] = None,
        daily_steps: int = 24,
        start_hour: float = 8.0,
    ) -> None:
        super().__init__(
            name=name,
            bus=bus,
            min_p_mw=min_p_mw,
            max_p_mw=max_p_mw,
            capacity=capacity,
            max_e_mwh=max_e_mwh,
            min_e_mwh=min_e_mwh,
            init_soc=init_soc,
            min_q_mvar=min_q_mvar,
            max_q_mvar=max_q_mvar,
            sn_mva=sn_mva,
            ch_eff=ch_eff,
            dsc_eff=dsc_eff,
            cost_curve_coefs=cost_curve_coefs,
            dt=dt,
        )

        # No trailing commas: these must be integers, not 1-element tuples.
        self.arrive_time = arrive_time
        self.depart_time = depart_time

        self.external_control = bool(external_control)
        self.action_callback = not self.external_control

        self.daily_steps = daily_steps
        self.start_hour = float(start_hour)
        if self.daily_steps <= 0:
            raise ValueError("daily_steps must be positive")

        if usage_type is None:
            upper = self.name.upper()
            if "RESIDENTIAL" in upper:
                usage_type = "residential"
            elif "OFFICE" in upper:
                usage_type = "office"
            elif "COMMERCIAL" in upper:
                usage_type = "commercial"
            else:
                usage_type = "generic"
        self.usage_type = str(usage_type).lower()

        self.connected = False

    def _hour_to_step(self, hour: float, next_day: bool = False) -> int:
        """
        Convert clock time to simulation step.

        Episode:
            t=0            -> 08:00 today
            t=daily_steps  -> 08:00 next day
        """
        hours_per_step = 24.0 / self.daily_steps

        relative_hour = hour - self.start_hour

        if relative_hour < 0:
            relative_hour += 24.0

        step = int(np.floor(relative_hour / hours_per_step))

        if next_day and step == 0:
            step = self.daily_steps

        return step

    def is_connected(self, step_idx: int) -> bool:
        """Return True when the EV is inside its daily parking window."""
        step_idx = step_idx % self.daily_steps
        arrival = self.arrive_time % self.daily_steps
        departure = self.depart_time % self.daily_steps

        if arrival == departure:
            return False

        if arrival < departure:
            return arrival <= step_idx < departure

        # Parking interval crosses midnight.
        return step_idx >= arrival or step_idx < departure

    def automatic_action(self) -> None:
        """Uncontrolled baseline: charge immediately at maximum feasible power."""
        if not self.connected or self.state.soc >= self.max_soc:
            self.action.c[:] = 0.0
            return

        max_energy_charge = (
            (self.max_soc - self.state.soc)
            * self.capacity
            / self.ch_eff
            / self.dt
        )
        self.action.c[0] = max(0.0, min(self.max_p_mw, max_energy_charge))

        if self.action.c.size > 1:
            self.action.c[1] = 0.0

    def update_state(self, step_idx: int = None) -> None:
        """Update connection state, charging action, and SOC."""
        step_idx = 0 if step_idx is None else step_idx

        self.connected = self.is_connected(step_idx)
        self.state.on = self.connected

        if not self.connected:
            self.state.P = 0.0
            self.state.Q = 0.0
            self.action.c[:] = 0.0
            return

        # Uncontrolled baseline charges automatically and remains physically feasible.
        if not self.external_control:
            self.automatic_action()

        # For RL control, do not modify the action here.
        # The agent is allowed to make poor charging decisions and receives
        # safety/reward penalties for violating SOC requirements.
        super().update_state(step_idx)

    def reset(self, *, rnd=None, init_soc: Optional[float] = None) -> None:
        rnd = np.random.default_rng() if rnd is None else rnd

        # The distributions below are deliberately simple and interpretable for
        # the EV paper.  Times are converted automatically for 24, 48, etc.
        if self.usage_type == "residential":
            if init_soc is None:
                init_soc = rnd.uniform(0.30, 0.60)
            self.arrive_time = ( # starting at 8:00 AM --> time: 0 
                rnd.integers(self._hour_to_step(17), self._hour_to_step(21) + 1)
            )
            self.depart_time = (
                rnd.integers(self._hour_to_step(6, next_day=True), self._hour_to_step(8, next_day=True) + 1)
            )

        elif self.usage_type == "office":
            if init_soc is None:
                init_soc = rnd.uniform(0.35, 0.65)
            self.arrive_time = (
                rnd.integers(self._hour_to_step(8.5), self._hour_to_step(10.5) + 1)
            )
            self.depart_time = (
                rnd.integers(self._hour_to_step(16.0), self._hour_to_step(18.5) + 1)
            )

        elif self.usage_type == "commercial":
            if init_soc is None:
                init_soc = rnd.uniform(0.30, 0.65)

            arrival_low = self._hour_to_step(9.0)
            arrival_high = self._hour_to_step(17.0)
            self.arrive_time = rnd.integers(arrival_low, arrival_high + 1)

            # Public/commercial vehicles have shorter dwell times.
            min_dwell = max(1, int(round(1.5 / 24.0 * self.daily_steps)))
            max_dwell = max(min_dwell + 1, int(round(4.0 / 24.0 * self.daily_steps)))
            dwell = rnd.integers(min_dwell, max_dwell + 1)
            self.depart_time = min(self.arrive_time + dwell, self.daily_steps - 1)

        elif init_soc is None:
            init_soc = rnd.uniform(self.min_soc, self.max_soc)

        # Keep initial SOC inside the EV's physical bounds.
        init_soc = np.clip(init_soc, self.min_soc, self.max_soc)
        super().reset(rnd=rnd, init_soc=init_soc)

        self.connected = self.is_connected(0)
        self.state.on = self.connected
        self.state.P = 0.0
        self.state.Q = 0.0
        if self.action.c.size:
            self.action.c[:] = 0.0
