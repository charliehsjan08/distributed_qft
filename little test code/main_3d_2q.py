import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from qiskit import QuantumCircuit, transpile
# Import synth_qft_full instead of QFT or QFTGate
from qiskit.synthesis.qft import synth_qft_full
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from src.circuits import create_registers, build_state_preparation
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion

def get_total_crossings(approx_degree):
    """Calculates the total cross-chip teleportation epochs for a given degree."""
    # Use the synthesis function directly to generate the circuit
    qc_perfect = synth_qft_full(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    
    unrolled = transpile(qc_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    
    cross_count = sum(1 for inst in unrolled.data if inst.operation.name == 'cx' and 
                      (unrolled.find_bit(inst.qubits[0]).index < 4) != (unrolled.find_bit(inst.qubits[1]).index < 4))
    return cross_count

def build_dd_qft(circuit, registers, approx_degree=4, dd_percentage=0.0, total_crossings=0, zne_scale=1):
    """
    Builds a pure Telegate distributed QFT.
    Applies Dynamical Decoupling (DD) to a percentage of the idle teleportation epochs.
    """
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    # Generate the approximated circuit here
    qc_perfect = synth_qft_full(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled_perfect = transpile(qc_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

    # Determine exactly how many epochs to protect based on the requested percentage
    num_dd_epochs = int(np.round(total_crossings * (dd_percentage / 100.0)))
    
    protection_map = ['DD'] * num_dd_epochs + ['None'] * (total_crossings - num_dd_epochs)
    np.random.shuffle(protection_map)

    for inst in unrolled_perfect.data:
        gate_name = inst.operation.name
        logical_qubits = [unrolled_perfect.find_bit(q).index for q in inst.qubits]

        if gate_name == 'cx':
            q_ctrl, q_targ = logical_qubits[0], logical_qubits[1]
            is_ctrl_on_A = q_ctrl < 4
            is_targ_on_A = q_targ < 4

            if is_ctrl_on_A != is_targ_on_A:
                phys_ctrl = get_physical_qubit(q_ctrl)
                phys_targ = get_physical_qubit(q_targ)
                
                # --- PURE TELEGATE EXECUTION ---
                circuit.reset(qr_comm_A)
                circuit.reset(qr_comm_B)
                circuit.h(qr_comm_A)
                circuit.cx(qr_comm_A, qr_comm_B)

                if is_ctrl_on_A:
                    circuit.cx(phys_ctrl, qr_comm_A)
                    circuit.cx(qr_comm_B, phys_targ)
                    circuit.h(qr_comm_B) 
                    circuit.measure(qr_comm_A, cr_tele[0])
                    circuit.measure(qr_comm_B, cr_tele[1])
                    with circuit.if_test((cr_tele[1], 1)): circuit.z(phys_ctrl)
                    with circuit.if_test((cr_tele[0], 1)): circuit.x(phys_targ)
                else:
                    circuit.cx(phys_ctrl, qr_comm_B)
                    circuit.cx(qr_comm_A, phys_targ)
                    circuit.h(qr_comm_A) 
                    circuit.measure(qr_comm_B, cr_tele[1])
                    circuit.measure(qr_comm_A, cr_tele[0])
                    with circuit.if_test((cr_tele[0], 1)): circuit.z(phys_ctrl)
                    with circuit.if_test((cr_tele[1], 1)): circuit.x(phys_targ)
                
                # --- DYNAMICAL DECOUPLING ON IDLE QUBITS ---
                idle_logical_indices = [i for i in range(8) if i not in (q_ctrl, q_targ)]
                epoch_strategy = protection_map.pop(0) if protection_map else 'None'
                
                epoch_depth_factor = (total_crossings - len(protection_map))
                variable_drift = (np.pi / 64) * epoch_depth_factor * zne_scale
                
                for idx in idle_logical_indices:
                    phys_idle = get_physical_qubit(idx)
                    
                    if epoch_strategy == 'DD':
                        circuit.rz(variable_drift / 2, phys_idle)
                        circuit.x(phys_idle)
                        circuit.rz(variable_drift / 2, phys_idle)
                        circuit.x(phys_idle)
                    else:
                        circuit.rz(variable_drift, phys_idle)
                        
            else:
                phys_args = [get_physical_qubit(q) for q in logical_qubits]
                circuit.cx(*phys_args)
        else:
            phys_args = [get_physical_qubit(q) for q in logical_qubits]
            circuit.append(inst.operation, phys_args)

def run_error_sweep(error_multipliers=[0.5, 1.0, 2.0]):
    """
    Sweeps through different hardware noise environments and 
    generates a figure for each.
    """
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    # Generate the exact circuit for ideal uncomputation
    qft_exact_qc = synth_qft_full(num_qubits=8, approximation_degree=0, do_swaps=False)
    inv_gate = UnitaryGate(Operator(qft_exact_qc.inverse().compose(V_prep.inverse())).data, label="Ideal_Uncompute")

    # ... (the rest of your sweep logic remains exactly the same)

    degrees = [0, 4, 7]
    dd_pcts = [0, 50, 100]

    for multiplier in error_multipliers:
        print(f"\n--- Running Sweep for Error Multiplier: {multiplier}x ---")
        fidelity_matrix = np.zeros((len(degrees), len(dd_pcts)))

        for i, deg in enumerate(degrees):
            total_cross = get_total_crossings(deg)
            for j, pct in enumerate(dd_pcts):
                
                m3_fidelities = []
                for scale in [1, 3, 5]:
                    circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
                    circ.compose(V_prep, qubits=data_qubits, inplace=True)
                    
                    build_dd_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), 
                                 approx_degree=deg, dd_percentage=pct, total_crossings=total_cross, zne_scale=scale)
                    
                    circ.append(inv_gate, data_qubits)
                    circ.measure(data_qubits, cr_out)

                    scaled_model = get_comprehensive_noise_model(
                        t1_us=150.0 / (scale * multiplier), 
                        t2_us=100.0 / (scale * multiplier), 
                        pulse_err_1q=0.0005 * scale * multiplier,
                        pulse_err_2q=0.001 * scale * multiplier,
                        spam_error=0.01 * scale * multiplier
                    )
                    
                    counts = run_local_simulation(circ, noise_model=scaled_model, shots=2000)
                    m3_fidelities.append(apply_matrix_inversion(counts, 2000, p_error=0.01 * scale * multiplier))
                
                fidelity_matrix[i, j] = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8

        plt.figure(figsize=(8, 6))
        plt.imshow(fidelity_matrix, cmap='viridis', origin='lower')
        plt.colorbar(label='Fidelity')
        plt.title(f'Noise Multiplier: {multiplier}x')
        plt.xticks(range(len(dd_pcts)), [f"{p}%" for p in dd_pcts])
        plt.yticks(range(len(degrees)), degrees)
        plt.savefig(f'Sweep_Error_{multiplier}x.png')
        plt.close()

if __name__ == "__main__":
    run_error_sweep()