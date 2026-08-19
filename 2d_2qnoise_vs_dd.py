import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress the QFT deprecation warning for clean logs
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    return sum(1 for inst in unrolled.data if inst.operation.name == 'cx' and 
               (unrolled.find_bit(inst.qubits[0]).index < 4) != (unrolled.find_bit(inst.qubits[1]).index < 4))

def build_dd_sweep_qft(circuit, registers, approx_degree=4, dd_ratio=0.0, total_crossings=0, zne_scale=1, phase_drift_per_epoch=0.0):
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

    num_dd_epochs = int(round(dd_ratio * total_crossings))
    protection_map = ['DD'] * num_dd_epochs + ['None'] * (total_crossings - num_dd_epochs)
    if len(protection_map) > 0:
        np.random.shuffle(protection_map)

    # REPAIRED LOOP: Iterating through the transpiled circuit
    for inst in unrolled.data:
        gate_name = inst.operation.name
        logical_qubits = [unrolled.find_bit(q).index for q in inst.qubits]

        if gate_name == 'cx':
            q_ctrl, q_targ = logical_qubits[0], logical_qubits[1]
            is_ctrl_on_A = q_ctrl < 4
            is_targ_on_A = q_targ < 4

            if is_ctrl_on_A != is_targ_on_A:
                phys_ctrl = get_physical_qubit(q_ctrl)
                phys_targ = get_physical_qubit(q_targ)
                
                # Full Telegate Routing Bridge
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
                
                # --- DD PROTECTION ---
                if len(protection_map) > 0:
                    strat = protection_map.pop(0)
                    epoch_depth_factor = (total_crossings - len(protection_map))
                    
                    # Tie phase drift to the 2Q error so the 0.0% baseline is perfectly ideal
                    variable_drift = phase_drift_per_epoch * epoch_depth_factor * zne_scale
                    
                    idle_indices = [i for i in range(8) if i not in (q_ctrl, q_targ)]
                    for idx in idle_indices:
                        phys = get_physical_qubit(idx)
                        if strat == 'DD':
                            # REPAIRED: Even number of X gates ensures logical identity
                            # while physically reversing phase drift
                            circuit.rz(variable_drift / 2, phys)
                            circuit.x(phys)
                            circuit.rz(variable_drift / 2, phys)
                            circuit.x(phys)
                        else:
                            # Unprotected idle epoch
                            circuit.rz(variable_drift, phys)
            else: 
                circuit.cx(get_physical_qubit(q_ctrl), get_physical_qubit(q_targ))
        else: 
            circuit.append(inst.operation, [get_physical_qubit(q) for q in logical_qubits])

def execute_noise_vs_dd_sweep(TEST_DEGREE):
    print("--- Mapping 2D Phase Diagram: DD Ratio vs. 2Q Gate Error ---")
    
    total_crossings = get_total_crossings(TEST_DEGREE)
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    ideal_inverse = QuantumCircuit(8)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    inv_gate = UnitaryGate(Operator(ideal_inverse).data, label="Ideal_Uncompute")

    dd_ratios = np.linspace(0.0, 1.0, 5) # 0%, 25%, 50%, 75%, 100%
    gate_errors = np.arange(0.0, 0.026, 0.005) # 0.0% to 2.5% in 0.5% steps
    
    fidelity_matrix = np.zeros((len(gate_errors), len(dd_ratios)))

    for i, err in enumerate(gate_errors):
        print(f"\n--- Testing 2Q Gate Error Rate at {err*100:.2f}% ---")
        
        # Calculate artificial phase drift proportional to 2Q error (0.0% error = 0 drift)
        current_phase_drift = (np.pi / 48) * (err / 0.01) if err > 0 else 0.0
        
        for j, ratio in enumerate(dd_ratios):
            print(f"  DD Ratio: {int(ratio*100)}% -> ", end="", flush=True)
            m3_fidelities = []
            for scale in [1, 3, 5]:
                circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
                circ.compose(V_prep, qubits=data_qubits, inplace=True)
                circ.barrier()
                
                # Pass the phase drift down to the builder
                build_dd_sweep_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), 
                                  TEST_DEGREE, ratio, total_crossings, scale, current_phase_drift)
                                  
                circ.barrier()
                circ.append(inv_gate, data_qubits)
                circ.measure(data_qubits, cr_out)

                model = get_comprehensive_noise_model(t1_us=np.inf, t2_us=np.inf, pulse_err_1q=0.0, 
                                                     pulse_err_2q=err*scale, spam_error=0.0)
                
                counts = run_local_simulation(circ, noise_model=model, shots=3000)
                m3_fid = apply_matrix_inversion(counts, 3000, 0.0)
                m3_fidelities.append(m3_fid)
            
            mitigated_fid = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8
            final_mitigated = min(max(mitigated_fid, 0), 1.0)
            
            fidelity_matrix[i, j] = final_mitigated
            print(f"{final_mitigated*100:.1f}%")

    # Plot generation
    plt.figure(figsize=(10, 8))
    im = plt.imshow(fidelity_matrix * 100, cmap='plasma', aspect='auto', origin='lower',
                    extent=[-12.5, 112.5, gate_errors[0]*100-0.25, gate_errors[-1]*100+0.25], vmin=0, vmax=100)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Mitigated Output Fidelity (%)', rotation=270, labelpad=20)
    
    plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    plt.yticks(gate_errors * 100)
    plt.xlabel('Dynamical Decoupling Protection Ratio (%)')
    plt.ylabel('Physical 2Q Gate Error (%)')
    plt.title(f'Phase Diagram: Degree {TEST_DEGREE} Noise Tolerance')
    
    for idx_y in range(len(gate_errors)):
        for idx_x in range(len(dd_ratios)):
            val = fidelity_matrix[idx_y, idx_x] * 100
            text_color = "black" if val > 70 else "white"
            plt.text(dd_ratios[idx_x]*100, gate_errors[idx_y]*100, f'{val:.1f}', 
                     ha="center", va="center", color=text_color, fontsize=10)
                     
    plt.savefig('noise_vs_dd_sweep_degree'+str(TEST_DEGREE)+'.png', dpi=300)
    print("\nSweep complete. Saved visualization to 'noise_vs_dd_sweep.png'.")

if __name__ == "__main__":
    for degree in range(8):
        execute_noise_vs_dd_sweep(TEST_DEGREE=degree)