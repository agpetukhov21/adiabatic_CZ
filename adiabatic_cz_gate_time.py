#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:59:35 2020
"""

"""Example showing coupler pulse shaping for the CZ-gate on gmon system."""


import numpy as np
from matplotlib import pyplot as plt
from adiabatic_cz_utils import cz_schedule
from pulse_shaping.contrib.calculations.examples_util import FSimGate, in_basis_optimized
from pulse_shaping.contrib.calculations.math_utils import sub_eigensystem
from pulse_shaping.contrib.calculations.system_examples import calibrated_gmon_system
from pulse_shaping.contrib.calculations.system_utils import build_qubit
from pulse_shaping.propagator.propagator import solve_unitary
from pulse_shaping.schedules.schedules import change_of_variables
from pulse_shaping.unitary_analysis.unitary_decomposition import decompose_fsim, FSimAngles
from pulse_shaping.utils.util import dressed_basis

# 1st Good case
# pulse_shape (1 - 16 * x**2 * (1 - x)**2)**3.5
# approach_frequency=6.0335
# g_QC_swap_GHz = 0.26
# t_pulse=31.1614

# 2nd Good case
# pulse_shape (1 - 16 * x**2 * (1 - x)**2)**4.5
# approach_frequency=6.06
# g_QC_swap_GHz = 0.24
# t_pulse = 33.1015



# Phase 1 - Construct the circuit Hamiltonian object.
# Specify the qubit circuits

# Circuit parameters
circuit_params = dict(EC_h_GHz=0.22, EJ_max_h_GHz=30.0)
q0 = build_qubit('q0', 6.0, 3, circuit_params)
q1 = build_qubit('q1', 5.5, 3, circuit_params)
nulled_states = ((1, 1), (0, 2))

target_gate = FSimGate(swap_angle=0, CZ_phase=np.pi)

# create a system of 2 qubits + coupler, where coupler idle frequency is set
# to cancel out the energy splitting of the nulled states. q0_f10 and q1_f10 are
# the qubit biases that bring the nulled states on resonance.

g_QQ_swap_GHz = 0.016
g_QC_swap_GHz = 0.26

system, q0_f10, q1_f10 = calibrated_gmon_system((q0, q1),
                                                nulled_states=nulled_states,
                                                g_QQ_swap_GHz=g_QQ_swap_GHz,
                                                g_QC_swap_GHz=g_QC_swap_GHz,
                                                coupler_parameters=dict(
                                                    EC_h_GHz=0.22,
                                                    EJ_max_h_GHz=125),
                                                num_coupler_levels=2)
cp = system[2]

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def plot_nulled_states():
    # confirm that hybridization is small near interaction frequency

    lower = max(7.5, cp.idle_frequency - 1)
    upper = min(cp.interpolator().max_f10_GHz, cp.idle_frequency + 1)
    cp_f10_range = np.linspace(lower, upper, 100)
    biases = {q0.label: q0_f10, q1.label: q1_f10, cp.label: cp_f10_range}

    ham = system.build_ham()

    evals, evecs = np.linalg.eigh(ham.compute(biases))

    nulled_state_with_cp = [s + (0,) for s in nulled_states]
    sub_evals, _ = sub_eigensystem(evals, evecs,
                                   system.local_dimensions,
                                   nulled_state_with_cp)

# Phase 2 - Specify the schedule
schedule = cz_schedule(*(q.label for q in system))
schedule = change_of_variables(
    schedule,
    {('hold_f10_0', 'hold_f10_1'):
         lambda f_int, f_detune: (f_int - f_detune / 2, f_int + f_detune / 2)}
)


approach_frequency = 6.0335
tmin = 0.01
tmax = 40.01
num_int_points = 1000

list_leakage_error = []
pulse_times_size = 1000
pulse_times = np.linspace(tmin, tmax, pulse_times_size)

for t_pulse in pulse_times:   

    f_avg_at_res = (q0_f10 + q1_f10) / 2
    params = dict(f_int=f_avg_at_res,
              f_detune=q1_f10 - q0_f10,
              hold_f10_cp=approach_frequency,
              idle_f10_0=q0.idle_frequency,
              idle_f10_1=q1.idle_frequency,
              idle_f10_cp=cp.idle_frequency,
              t_cp_delay=0,
              t_hold=t_pulse,
              t_ramp=0,
              t_pad=0)
    max_idle_GHz = max(q0.idle_frequency, q1.idle_frequency)



# Parameterize the propagator as a function of the schedule parameters

    def propagator(schedule_params, num_int_points, ham):
        times = np.linspace(0, schedule.pulse_duration(schedule_params),
                        num_int_points)
        f10_funs = schedule.build_pulses(schedule_params)
    # apply gaussian filter to pulse shapes
    #f10_funs = {
    #    tmon: gaussian_filter_interp(f10_fun, rise_time_ns, (0, times[-1]))
    #   for tmon, f10_fun in f10_funs.items()}

        def ham_fun(t):
            return ham.compute({tmon: f10_fun(t)
                            for tmon, f10_fun in f10_funs.items()})

        return solve_unitary(ham_fun, times)

    ham = system.build_ham()
# dressed computational basis used to compute fidelity
    d_basis = dressed_basis(
    # idle hamiltonian
    ham.compute({q.label: q.idle_frequency for q in system}),
    basis_dims=(2, 2, 1),
    full_dims=system.local_dimensions)

    duration = schedule.pulse_duration(params)
    times = np.linspace(0, duration, num_int_points)

    fitted_prop = propagator(params, num_int_points, ham)
    fitted_prop = in_basis_optimized(fitted_prop, d_basis)
    fitted_angles = decompose_fsim(fitted_prop)
    fitted_anglesf = decompose_fsim(fitted_prop[-1])

    list_leakage_error.append([fitted_anglesf[FSimAngles.LEAKAGE],np.cos(fitted_anglesf[FSimAngles.CZ]/2)**2,10**(-4)])

#%%
plt.figure()
plt.semilogy(pulse_times,list_leakage_error)
plt.xlabel('pulse time (ns)')
plt.ylabel('leakage error')
plt.show()
#    ylim = plt.ylim()
#    plt.plot([cp.idle_frequency] * 2, ylim, '--')
#    plt.ylim(ylim)
#    plt.tight_layout()

