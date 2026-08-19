"""
qft_experiment_harness.py
Master framework for testing distributed QFT architectures under various noise models.
Separates state preparation, test circuit execution, ideal uncomputation, and ZNE.
"""

import warnings
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator

# Ensure your local src modules are accessible
from src.circuits import create_registers, build_state_preparation
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion

# Suppress warnings for clean execution logs
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_total_crossings(approx_degree):
    """Utility function to calculate total cross-chip teleportation epochs."""
    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    return sum(1 for inst in unrolled.data if inst.operation.name == 'cx' and 
               (unrolled.find_bit(inst.qubits[0]).index < 4) != (unrolled.find_bit(inst.qubits[1]).index < 4))


# ==========================================
# STAGE 1: Random State Preparation
# ==========================================
def apply_state_preparation(circuit, data_qubits):
    """
    Creates and applies the random initial state.
    Returns the V_prep circuit so it can be perfectly inverted later.
    """
    v_prep = build_state_preparation(data_qubits)
    circuit.compose(v_prep, qubits=data_qubits, inplace=True)
    return v_prep


# ==========================================
# STAGE 2: Test Circuits (The "Red Ring")
# ==========================================
def build_test_circuit_dd(circuit, registers, approx_degree=4, dd_ratio=0.0, total_crossings=0, zne_scale=1, phase_drift_per_epoch=0.0):
    """Builds the distributed QFT with controllable Dynamical Decoupling coverage and phase drift."""
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

    num_dd_epochs = int(round(dd_ratio * total_crossings))
    protection_map = ['DD'] * num_dd_epochs + ['None'] * (total_crossings - num_dd_epochs)
    if len(protection_map) > 0:
        np.random.shuffle(protection_map)

    for inst in unrolled.data:
        gate_name = inst.operation.name
        logical_qubits = [unrolled.find_bit(q).index for q in inst.qubits]

        if gate_name == 'cx':
            q_ctrl, q_targ = logical_qubits[0], logical_qubits[1]
            if (q_ctrl < 4) != (q_targ < 4):
                phys_ctrl, phys_targ = get_physical_qubit(q_ctrl), get_physical_qubit(q_targ)
                
                # Standard Telegate Bridge
                circuit.reset(qr_comm_A); circuit.reset(qr_comm_B)
                circuit.h(qr_comm_A); circuit.cx(qr_comm_A, qr_comm_B)
                
                if q_ctrl < 4:
                    circuit.cx(phys_ctrl, qr_comm_A); circuit.cx(qr_comm_B, phys_targ); circuit.h(qr_comm_B)
                    circuit.measure(qr_comm_A, cr_tele[0]); circuit.measure(qr_comm_B, cr_tele[1])
                    with circuit.if_test((cr_tele[1], 1)): circuit.z(phys_ctrl)
                    with circuit.if_test((cr_tele[0], 1)): circuit.x(phys_targ)
                else:
                    circuit.cx(phys_ctrl, qr_comm_B); circuit.cx(qr_comm_A, phys_targ); circuit.h(qr_comm_A)
                    circuit.measure(qr_comm_B, cr_tele[1]); circuit.measure(qr_comm_A, cr_tele[0])
                    with circuit.if_test((cr_tele[0], 1)): circuit.z(phys_ctrl)
                    with circuit.if_test((cr_tele[1], 1)): circuit.x(phys_targ)
                
                # Idle Qubit Phase Drift and DD Protection
                if len(protection_map) > 0:
                    strat = protection_map.pop(0)
                    variable_drift = phase_drift_per_epoch * (total_crossings - len(protection_map)) * zne_scale
                    for idx in [i for i in range(8) if i not in (q_ctrl, q_targ)]:
                        phys = get_physical_qubit(idx)
                        if strat == 'DD':
                            circuit.rz(variable_drift / 2, phys); circuit.x(phys)
                            circuit.rz(variable_drift / 2, phys); circuit.x(phys)
                        else:
                            circuit.rz(variable_drift, phys)
            else: 
                circuit.cx(get_physical_qubit(q_ctrl), get_physical_qubit(q_targ))
        else: 
            circuit.append(inst.operation, [get_physical_qubit(q) for q in logical_qubits])

def build_test_circuit_hybrid(circuit, registers, approx_degree=4, num_locc=0, total_crossings=0, zne_scale=1):
    """Builds the distributed QFT integrating ideal LOCC cuts alongside physical Telegates."""
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

    method_map = ['LOCC'] * min(num_locc, total_crossings) + ['Telegate'] * (total_crossings - min(num_locc, total_crossings))
    np.random.shuffle(method_map)

    for inst in unrolled.data:
        gate_name = inst.operation.name
        logical_qubits = [unrolled.find_bit(q).index for q in inst.qubits]

        if gate_name == 'cx':
            q_ctrl, q_targ = logical_qubits[0], logical_qubits[1]
            if (q_ctrl < 4) != (q_targ < 4):
                phys_ctrl, phys_targ = get_physical_qubit(q_ctrl), get_physical_qubit(q_targ)
                
                if method_map.pop(0) == 'Telegate':
                    circuit.reset(qr_comm_A); circuit.reset(qr_comm_B)
                    circuit.h(qr_comm_A); circuit.cx(qr_comm_A, qr_comm_B)
                    if q_ctrl < 4:
                        circuit.cx(phys_ctrl, qr_comm_A); circuit.cx(qr_comm_B, phys_targ); circuit.h(qr_comm_B)
                        circuit.measure(qr_comm_A, cr_tele[0]); circuit.measure(qr_comm_B, cr_tele[1])
                        with circuit.if_test((cr_tele[1], 1)): circuit.z(phys_ctrl)
                        with circuit.if_test((cr_tele[0], 1)): circuit.x(phys_targ)
                    else:
                        circuit.cx(phys_ctrl, qr_comm_B); circuit.cx(qr_comm_A, phys_targ); circuit.h(qr_comm_A)
                        circuit.measure(qr_comm_B, cr_tele[1]); circuit.measure(qr_comm_A, cr_tele[0])
                        with circuit.if_test((cr_tele[0], 1)): circuit.z(phys_ctrl)
                        with circuit.if_test((cr_tele[1], 1)): circuit.x(phys_targ)
                else:
                    # LOCC bypass uses a mathematically perfect Unitary matrix (0% Hardware Noise)
                    cx_matrix = Operator([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).data
                    circuit.append(UnitaryGate(cx_matrix, label="LOCC_CX"), [phys_ctrl, phys_targ])
            else:
                circuit.cx(get_physical_qubit(q_ctrl), get_physical_qubit(q_targ))
        else:
            circuit.append(inst.operation, [get_physical_qubit(q) for q in logical_qubits])


# ==========================================
# STAGE 3: Exact Uncompute & Measure
# ==========================================
def apply_ideal_uncompute_and_measure(circuit, data_qubits, cr_out, v_prep):
    """
    Applies the mathematically perfect QFT^-1 and V_prep^-1 as a single unitary,
    preventing the simulator from injecting arbitrary gate noise into this stage.
    """
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    ideal_inverse_circ = QuantumCircuit(8)
    ideal_inverse_circ.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse_circ.compose(v_prep.inverse(), inplace=True)
    
    inv_gate = UnitaryGate(Operator(ideal_inverse_circ).data, label="Exact_Uncompute")
    circuit.append(inv_gate, data_qubits)
    circuit.measure(data_qubits, cr_out)


# ==========================================
# STAGE 4: Hardware Execution & ZNE
# ==========================================
def execute_noisy_simulation(circuit, shots, t1_us, t2_us, err_1q, err_2q, spam_err, mitigation_target_err):
    """Executes a single simulation run and applies readout mitigation if requested."""
    noise_model = get_comprehensive_noise_model(
        t1_us=t1_us, t2_us=t2_us, 
        pulse_err_1q=err_1q, pulse_err_2q=err_2q, spam_error=spam_err
    )
    counts = run_local_simulation(circuit, noise_model=noise_model, shots=shots)
    return apply_matrix_inversion(counts, shots, p_error=mitigation_target_err)

def apply_richardson_extrapolation(f1, f3, f5):
    """Geometrically extrapolates the Zero-Noise limit using 1x, 3x, and 5x scaled results."""
    mitigated_fid = (15 * f1 - 10 * f3 + 3 * f5) / 8
    return min(max(mitigated_fid, 0.0), 1.0)


# ==========================================
# EXPERIMENT CONTROLLER (For External Import)
# ==========================================
def get_zne_mitigated_fidelity(
    test_degree=4, 
    shots=3000, 
    builder_func=None, 
    builder_kwargs=None,
    t1_us=np.inf, 
    t2_us=np.inf, 
    err_1q=0.001, 
    err_2q=0.01, 
    spam_err=0.02,
    mitigation_target_err=None
):
    """
    Executes a complete ZNE pipeline for a specific hardware and architectural configuration.
    
    Args:
        test_degree (int): AQFT approximation degree.
        shots (int): Number of measurement shots per circuit.
        builder_func (callable): The function used to build Stage 2 (e.g., build_test_circuit_dd).
        builder_kwargs (dict): Specific parameters for the builder (e.g., {'dd_ratio': 0.5}).
        t1_us, t2_us, err_1q, err_2q, spam_err: Base hardware noise parameters.
        mitigation_target_err: SPAM mitigation parameter. Defaults to spam_err if None.
        
    Returns:
        float: The final ZNE mitigated fidelity (0.0 to 1.0).
    """
    if builder_func is None:
        raise ValueError("A builder_func (e.g., build_test_circuit_dd) must be provided.")
    
    if builder_kwargs is None:
        builder_kwargs = {}
        
    if mitigation_target_err is None:
        mitigation_target_err = spam_err
        
    total_crossings = get_total_crossings(test_degree)
    m3_fidelities = []
    
    # Run the ZNE loop (Scales: 1, 3, 5)
    for scale in [1, 3, 5]:
        
        qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
        data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
        circuit = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
        
        # [STAGE 1]
        v_prep = apply_state_preparation(circuit, data_qubits)
        circuit.barrier()
        
        # [STAGE 2] Dynamically inject the correct scale and architecture details
        builder_kwargs['approx_degree'] = test_degree
        builder_kwargs['total_crossings'] = total_crossings
        if 'zne_scale' in builder_func.__code__.co_varnames:
            builder_kwargs['zne_scale'] = scale
            
        builder_func(
            circuit=circuit, 
            registers=(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out),
            **builder_kwargs
        )
        circuit.barrier()
        
        # [STAGE 3]
        apply_ideal_uncompute_and_measure(circuit, data_qubits, cr_out, v_prep)
        
        # [STAGE 4] Execute with scaled noise
        fidelity = execute_noisy_simulation(
            circuit=circuit, shots=shots,
            t1_us=t1_us / scale if not np.isinf(t1_us) else np.inf, 
            t2_us=t2_us / scale if not np.isinf(t2_us) else np.inf,
            err_1q=err_1q * scale, 
            err_2q=err_2q * scale, 
            spam_err=spam_err * scale,
            mitigation_target_err=mitigation_target_err * scale
        )
        m3_fidelities.append(fidelity)
        
    # Extrapolate Final Data Point
    return apply_richardson_extrapolation(*m3_fidelities)