from typing import Optional

from .base import Device


class Switch(Device):
    """Controllable bus-bus switch.

    The switch is represented in pandapower as a bus-bus switch (``et='b'``).
    GridBaseEnv is responsible for writing ``state.closed`` to
    ``net.switch.closed`` before each power-flow calculation.

    Discrete action
    ---------------
    0 : open
    1 : closed
    """

    def __init__(
        self,
        name: str,
        fbus: str,
        tbus: str,
        *,
        closed: bool = True,
        switching_cost: float = 0.0,
    ) -> None:
        super().__init__()

        self.type = "SW"
        self.name = name
        self.fbus = fbus
        self.tbus = tbus
        self.switching_cost = switching_cost

        self._initial_closed = closed
        self.state.closed = self._initial_closed
        self._last_closed = self.state.closed

        self.set_action_space()

    def set_action_space(self) -> None:
        # Binary discrete action: 0 = open, 1 = closed.
        self.action.ncats = 2
        self.action.dim_d = 1
        self.action.sample()

    def update_state(self, step_idx: Optional[int] = None) -> None:
        if self.action.d.size:
            self.state.closed = bool(int(self.action.d[0]))

    def update_cost_safety(self) -> None:
        changed = int(self.state.closed != self._last_closed)
        self.cost = float(self.switching_cost * changed)
        self.safety = 0.0
        self._last_closed = self.state.closed

    def reset(self, rnd=None) -> None:
        self.state.closed = self._initial_closed
        self._last_closed = self.state.closed
        self.cost = 0.0
        self.safety = 0.0