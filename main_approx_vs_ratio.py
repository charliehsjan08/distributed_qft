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
    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    
    cross_count = sum(1 for inst in unrolled.data if inst.operation.name == 'cx' and 
                      (unrolled.find_bit(inst.qubits[0]).index < 4) != (unrolled.find_bit(inst.qubits[1]).index < 4))
    return cross_count

def build_hybrid_qft(circuit, registers, approx_degree=4, num_locc=0, total_crossings=0):
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled_perfect = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

    safe_locc_count = min(num_locc, total_crossings)
    method_map = ['LOCC'] * safe_locc_count + ['Telegate'] * (total_crossings - safe_locc_count)
    np.random.shuffle(method_map)

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
                
                selected_method = method_map.pop(0)

                if selected_method == 'Telegate':
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
                
                elif selected_method == 'LOCC':
                    cx_matrix = Operator([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).data
                    locc_gate = UnitaryGate(cx_matrix, label="LOCC_Virtual_CX")
                    circuit.append(locc_gate, [phys_ctrl, phys_targ])
            else:
                phys_args = [get_physical_qubit(q) for q in logical_qubits]
                circuit.cx(*phys_args)
        else:
            phys_args = [get_physical_qubit(q) for q in logical_qubits]
            circuit.append(inst.operation, phys_args)

def execute_2d_sweep():
    print("--- Executing 2D Optimization Sweep (Degree vs. LOCC) ---")
    
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    ideal_inverse = QuantumCircuit(8)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    inv_gate = UnitaryGate(Operator(ideal_inverse).data, label="Ideal_Uncompute")

    # Baseline Parameters
    base_err_1q = 0.001 
    base_err_2q = 0.01  
    base_spam = 0.02    
    base_num_shots = 5000 
    
    degrees_to_test = list(range(1, 8)) # Degrees 1 through 7
    max_crossings_overall = get_total_crossings(1) # Degree 1 will have the most crossings
    
    # Initialize the Heatmap Matrix with NaNs (Not-a-Number) to handle impossible combinations
    fidelity_matrix = np.full((len(degrees_to_test), max_crossings_overall + 1), np.nan)
    
    # Pre-calculate the crossings for each degree so we know our boundaries
    degree_crossings_map = {deg: get_total_crossings(deg) for deg in degrees_to_test}
    
    # Calculate Total Computations for sanity check
    total_sims = sum([crossings + 1 for crossings in degree_crossings_map.values()])
    print(f"Total Circuit Configurations to test: {total_sims} (Requires {total_sims * 3} M3 simulations). Grab a coffee!\n")

    for i, current_degree in enumerate(degrees_to_test):
        total_crossings = degree_crossings_map[current_degree]
        
        for current_locc_count in range(total_crossings + 1):
            print(f"Simulating -> Degree {current_degree} | LOCC Cuts: {current_locc_count}/{total_crossings} ", end="", flush=True)
            
            m3_fidelities = []
            for scale in [1, 3, 5]:
                circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
                circ.compose(V_prep, qubits=data_qubits, inplace=True)
                circ.barrier()
                
                build_hybrid_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), 
                                 approx_degree=current_degree, num_locc=current_locc_count, total_crossings=total_crossings)
                
                circ.barrier()
                circ.append(inv_gate, data_qubits)
                circ.measure(data_qubits, cr_out)

                scaled_model = get_comprehensive_noise_model(
                    t1_us=np.inf, t2_us=np.inf, 
                    pulse_err_1q=base_err_1q * scale, 
                    pulse_err_2q=base_err_2q * scale, 
                    spam_error=base_spam * scale
                )
                
                counts = run_local_simulation(circ, noise_model=scaled_model, shots=base_num_shots)
                m3_fid = apply_matrix_inversion(counts, base_num_shots, p_error=base_spam * scale)
                m3_fidelities.append(m3_fid)
                
            mitigated_fid = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8
            final_mitigated = min(max(mitigated_fid, 0), 1.0)
            
            # Store result in the matrix
            fidelity_matrix[i, current_locc_count] = final_mitigated
            print(f"-> Fidelity: {final_mitigated * 100:.2f}%")

    # --- Plotting the 2D Heatmap ---
    plt.figure(figsize=(12, 8))
    
    # Custom colormap: Dark Purple (low) to Bright Yellow (high)
    cmap = plt.cm.magma
    cmap.set_bad(color='#f0f0f0') # Set the impossible configurations to a light gray
    
    # Draw the Heatmap
    im = plt.imshow(fidelity_matrix * 100, cmap=cmap, aspect='auto', origin='lower', 
                    extent=[-0.5, max_crossings_overall + 0.5, degrees_to_test[0] - 0.5, degrees_to_test[-1] + 0.5])
    
    cbar = plt.colorbar(im)
    cbar.set_label('Mitigated Output Fidelity (%)', rotation=270, labelpad=20)
    
    plt.title('2D Parameter Optimization: AQFT Degree vs. Hybrid Routing')
    plt.xlabel('Number of Crossings using LOCC Bypass')
    plt.ylabel('Approximation Degree')
    
    plt.xticks(range(max_crossings_overall + 1))
    plt.yticks(degrees_to_test)
    
    # Overlay the exact fidelity numbers on top of the colored blocks
    for i in range(len(degrees_to_test)):
        for j in range(max_crossings_overall + 1):
            if not np.isnan(fidelity_matrix[i, j]):
                val = fidelity_matrix[i, j] * 100
                text_color = "black" if val > 80 else "white" # High contrast text
                plt.text(j, degrees_to_test[i], f'{val:.1f}', ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    plt.savefig('2D_Optimization_Landscape.png', dpi=300)
    print("\nSweep Complete! Check '2D_Optimization_Landscape.png'.")

if __name__ == "__main__":
    execute_2d_sweep()