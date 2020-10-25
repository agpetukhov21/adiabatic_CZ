from pulse_shaping.schedules.pulses import piecewise_linear
from pulse_shaping.schedules.schedules import Schedule, build_schedule


def cz_schedule(q0_label: str, q1_label: str, cp_label: str) -> Schedule:
    """Construct a coupler polynomial pulse schedule for the CZ gate.
       The qubit frequencies are fixed at their idle positions

    Args:
        q0_label: First qubit label.
        q1_label: Second qubit label.
        cp_label: Coupler label.

    """

    def q0_f10(time, t_pad: float, t_ramp: float, t_hold: float,
               idle_f10_0: float, hold_f10_0: float):
        return piecewise_linear(
            time, [t_pad,
                   t_pad + t_ramp,
                   t_pad + t_ramp + t_hold,
                   t_pad + 2 * t_ramp + t_hold],
            [idle_f10_0, idle_f10_0, idle_f10_0, idle_f10_0])

    def q1_f10(time, t_pad: float, t_ramp: float, t_hold: float,
               idle_f10_1: float, hold_f10_1: float):
        return piecewise_linear(
            time, [t_pad,
                   t_pad + t_ramp,
                   t_pad + t_ramp + t_hold,
                   t_pad + 2 * t_ramp + t_hold],
            [idle_f10_1, idle_f10_1, idle_f10_1, idle_f10_1])

    def cp_f10(time, t_hold: float, idle_f10_cp: float, hold_f10_cp: float):
        return hold_f10_cp + (idle_f10_cp - hold_f10_cp) * polynomial_shape(time, t_hold) 
    
    def polynomial_shape(t:float, t_total:float) -> float:
        x = t/t_total
        return (1 - 16 * x**2 * (1 - x)**2)**3.5

    def duration(t_pad: float, t_ramp: float, t_hold: float):
        return 2 * t_pad + 2 * t_ramp + t_hold

    return build_schedule(
        {q0_label: q0_f10, q1_label: q1_f10, cp_label: cp_f10},
        duration)


def cz_schedule_backup(q0_label: str, q1_label: str, cp_label: str) -> Schedule:
    """Construct a cascaded pulse schedule for the CZ gate.

    The qubits are brought on resonance prior to the coupler introducing
    the coupling.

    Args:
        q0_label: First qubit label.
        q1_label: Second qubit label.
        cp_label: Coupler label.

    """

    def q0_f10(time, t_pad: float, t_ramp: float, t_hold: float,
               idle_f10_0: float, hold_f10_0: float):
        return piecewise_linear(
            time, [t_pad,
                   t_pad + t_ramp,
                   t_pad + t_ramp + t_hold,
                   t_pad + 2 * t_ramp + t_hold],
            [idle_f10_0, idle_f10_0, idle_f10_0, idle_f10_0])

    def q1_f10(time, t_pad: float, t_ramp: float, t_hold: float,
               idle_f10_1: float, hold_f10_1: float):
        return piecewise_linear(
            time, [t_pad,
                   t_pad + t_ramp,
                   t_pad + t_ramp + t_hold,
                   t_pad + 2 * t_ramp + t_hold],
            [idle_f10_1, idle_f10_1, idle_f10_1, idle_f10_1])

    def cp_f10(time, t_pad: float, t_ramp: float, t_hold: float,
               t_cp_delay: float, idle_f10_cp: float, hold_f10_cp: float):
        return piecewise_linear(
            time, [t_pad + t_cp_delay,
                   t_pad + t_cp_delay + t_ramp,
                   t_pad - t_cp_delay + t_ramp + t_hold,
                   t_pad - t_cp_delay + 2 * t_ramp + t_hold],
            [idle_f10_cp, hold_f10_cp, hold_f10_cp, idle_f10_cp])

    def duration(t_pad: float, t_ramp: float, t_hold: float):
        return 2 * t_pad + 2 * t_ramp + t_hold

    return build_schedule(
        {q0_label: q0_f10, q1_label: q1_f10, cp_label: cp_f10},
        duration)

