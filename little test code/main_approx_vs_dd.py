import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator

from src.circuits import create_registers, build_state_preparation
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion

def get_total_crossings(approx_degree):
    """Calculates the total cross-chip teleportation epochs for a given degree."""
    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    
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

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled_perfect = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

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
                
                # The deeper into the circuit, the longer the wait time, the worse the phase drift
                epoch_depth_factor = (total_crossings - len(protection_map))
                variable_drift = (np.pi / 64) * epoch_depth_factor * zne_scale
                
                for idx in idle_logical_indices:
                    phys_idle = get_physical_qubit(idx)
                    
                    if epoch_strategy == 'DD':
                        # Spin Echo: reverses T2* drift, but pays physical X-gate noise tax
                        circuit.rz(variable_drift / 2, phys_idle)
                        circuit.x(phys_idle)
                        circuit.rz(variable_drift / 2, phys_idle)
                        circuit.x(phys_idle)
                    else:
                        # Suffers full coherent phase drift due to finite T2
                        circuit.rz(variable_drift, phys_idle)
                        
            else:
                phys_args = [get_physical_qubit(q) for q in logical_qubits]
                circuit.cx(*phys_args)
        else:
            phys_args = [get_physical_qubit(q) for q in logical_qubits]
            circuit.append(inst.operation, phys_args)


def execute_dd_landscape():
    print("--- Executing 2D DD Optimization Landscape ---")
    
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    ideal_inverse = QuantumCircuit(8)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    inv_gate = UnitaryGate(Operator(ideal_inverse).data, label="Ideal_Uncompute")

    # Standard Hardware Model (Requires heavy DD)
    base_err_1q = 0.0005 
    base_err_2q = 0.001  
    base_spam = 0.01    
    base_num_shots = 5000 
    
    # 2D Sweep Parameters
    degrees_to_test = list(range(0, 8)) # Y-axis: Degree 0 to 7
    dd_percentages_to_test = [0, 20, 40, 60, 80, 100] # X-axis: Normalized to % of idle epochs protected
    
    # Heatmap Matrix
    fidelity_matrix = np.full((len(degrees_to_test), len(dd_percentages_to_test)), np.nan)
    
    print(f"Total Circuit Configurations to test: {len(degrees_to_test) * len(dd_percentages_to_test)}.\n")

    for i, current_degree in enumerate(degrees_to_test):
        total_crossings = get_total_crossings(current_degree)
        
        for j, current_dd_pct in enumerate(dd_percentages_to_test):
            print(f"Simulating -> Degree {current_degree} | DD Coverage: {current_dd_pct}% ", end="", flush=True)
            
            m3_fidelities = []
            for scale in [1, 3, 5]:
                circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
                circ.compose(V_prep, qubits=data_qubits, inplace=True)
                circ.barrier()
                
                build_dd_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), 
                             approx_degree=current_degree, dd_percentage=current_dd_pct, 
                             total_crossings=total_crossings, zne_scale=scale)
                
                circ.barrier()
                circ.append(inv_gate, data_qubits)
                circ.measure(data_qubits, cr_out)

                # Standard T1/T2 times (50us) to force coherent phase drift
                scaled_model = get_comprehensive_noise_model(
                    t1_us=150.0 / scale, 
                    t2_us=100.0 / scale, 
                    pulse_err_1q=base_err_1q * scale, 
                    pulse_err_2q=base_err_2q * scale, 
                    spam_error=base_spam * scale
                )
                
                counts = run_local_simulation(circ, noise_model=scaled_model, shots=base_num_shots)
                m3_fid = apply_matrix_inversion(counts, base_num_shots, p_error=base_spam * scale)
                m3_fidelities.append(m3_fid)
                
            mitigated_fid = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8
            final_mitigated = min(max(mitigated_fid, 0), 1.0)
            
            fidelity_matrix[i, j] = final_mitigated
            print(f"-> Fidelity: {final_mitigated * 100:.1f}%")

    # --- Plotting the 2D Heatmap ---
    plt.figure(figsize=(10, 8))
    
    # Custom colormap: Dark Indigo (low) to Bright Yellow (high)
    cmap = plt.cm.magma
    
    im = plt.imshow(fidelity_matrix * 100, cmap=cmap, aspect='auto', origin='lower')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Mitigated Output Fidelity (%)', rotation=270, labelpad=20)
    
    plt.title('2D DD Optimization: AQFT Degree vs. Spin Echo Coverage')
    plt.xlabel('Idle Teleportation Epochs Protected by DD (%)')
    plt.ylabel('Approximation Degree')
    
    plt.xticks(range(len(dd_percentages_to_test)), [f"{p}%" for p in dd_percentages_to_test])
    plt.yticks(range(len(degrees_to_test)), degrees_to_test)
    
    # Overlay the exact fidelity numbers
    for i in range(len(degrees_to_test)):
        for j in range(len(dd_percentages_to_test)):
            val = fidelity_matrix[i, j] * 100
            text_color = "black" if val > 60 else "white"
            plt.text(j, i, f'{val:.1f}', ha="center", va="center", color=text_color, fontsize=10)

    plt.tight_layout()
    plt.savefig('2D_DD_Landscape.png', dpi=300)
    print("\nSweep Complete! Check '2D_DD_Landscape.png'.")

if __name__ == "__main__":
    execute_dd_landscape()