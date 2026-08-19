import numpy as np
from qiskit_aer.noise import depolarizing_error, NoiseModel, thermal_relaxation_error, ReadoutError

def get_comprehensive_noise_model(t1_us=100.0, t2_us=120.0, pulse_err_1q=0.001, pulse_err_2q=0.01, spam_error=0.03):
    """
    Generalized physics NoiseModel. 
    To isolate an error, set others to ideal (e.g., t1_us=np.inf, pulse_err_2q=0.0).
    """
    model = NoiseModel()
    
    # Timing constraints (microseconds)
    time_cx = 0.4
    time_single = 0.05
    
    # 1. Thermal relaxation
    if t1_us != np.inf and t2_us != np.inf:
        err_t1t2_1q = thermal_relaxation_error(t1_us, t2_us, time_single)
        err_t1t2_2q = thermal_relaxation_error(t1_us, t2_us, time_cx).expand(
                      thermal_relaxation_error(t1_us, t2_us, time_cx))
    else:
        err_t1t2_1q = None
        err_t1t2_2q = None

    # 2. Gate Pulse errors
    err_depol_1q = depolarizing_error(pulse_err_1q, 1) if pulse_err_1q > 0 else None
    err_depol_2q = depolarizing_error(pulse_err_2q, 2) if pulse_err_2q > 0 else None
    
    # Combine 1Q errors
    combined_1q = None
    if err_t1t2_1q and err_depol_1q: combined_1q = err_depol_1q.compose(err_t1t2_1q)
    elif err_t1t2_1q: combined_1q = err_t1t2_1q
    elif err_depol_1q: combined_1q = err_depol_1q

    # Combine 2Q errors
    combined_2q = None
    if err_t1t2_2q and err_depol_2q: combined_2q = err_depol_2q.compose(err_t1t2_2q)
    elif err_t1t2_2q: combined_2q = err_t1t2_2q
    elif err_depol_2q: combined_2q = err_depol_2q

    # Apply to model
    if combined_1q: model.add_all_qubit_quantum_error(combined_1q, ['x', 'sx', 'rz'])
    if combined_2q: model.add_all_qubit_quantum_error(combined_2q, ['cx'])
    
    # 3. SPAM readout errors
    if spam_error > 0:
        model.add_all_qubit_readout_error(ReadoutError([[1 - spam_error, spam_error], [spam_error, 1 - spam_error]]))
    
    return model

'''
    # 3. SPAM readout errors (Targeting ONLY the final 8 Data Qubits)
    if spam_error > 0:
        error_matrix = ReadoutError([[1 - spam_error, spam_error], [spam_error, 1 - spam_error]])
        # The data qubits are indices 0, 1, 2, 3 (Chip A) and 6, 7, 8, 9 (Chip B)
        terminal_qubits = [0, 1, 2, 3, 6, 7, 8, 9]
        for q in terminal_qubits:
            model.add_readout_error(error_matrix, [q])
'''

    