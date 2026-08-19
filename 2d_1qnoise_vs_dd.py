import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress the QFT deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator

from src.circuits import create_registers, build_state_preparation
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion

def get_total_crossings(approx_degree):
    """Calculates the total cross-chip teleportation epochs."""
    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    return sum(1 for inst in unrolled.data if inst.operation.name == 'cx' and 
               (unrolled.find_bit(inst.qubits[0]).index < 4) != (unrolled.find_bit(inst.qubits[1]).index < 4))

def build_dd_sweep_qft(circuit, registers, approx_degree=4, dd_ratio=0.0, total_crossings=0, zne_scale=1):
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

    num_dd_epochs = int(round(dd_ratio * total_crossings))
    protection_map = ['DD'] * num_dd_epochs + ['None'] * (total_crossings - num_dd_epochs)
    if len(protection_map) > 0: np.random.shuffle(protection_map)

    for inst in unrolled.data:
        gate_name = inst.operation.name
        logical_qubits = [unrolled.find_bit(q).index for q in inst.qubits]

        if gate_name == 'cx':
            q_ctrl, q_targ = logical_qubits[0], logical_qubits[1]
            if (q_ctrl < 4) != (q_targ < 4):
                phys_ctrl, phys_targ = get_physical_qubit(q_ctrl), get_physical_qubit(q_targ)
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
                
                # --- DD PROTECTION (1Q pulses) ---
                if len(protection_map) > 0 and protection_map.pop(0) == 'DD':
                    for idx in range(8):
                        if idx != q_ctrl and idx != q_targ:
                            # Perfect X-X sequence for DD (Identity in noise-free, Error-injecting in noisy)
                            circuit.x(get_physical_qubit(idx))
                            circuit.x(get_physical_qubit(idx))
            else:
                circuit.cx(get_physical_qubit(q_ctrl), get_physical_qubit(q_targ))
        else:
            circuit.append(inst.operation, [get_physical_qubit(q) for q in logical_qubits])

def execute_noise_vs_dd_sweep(TEST_DEGREE):
    print(f"\n--- Processing Approximation Degree {TEST_DEGREE} ---")
    
    total_crossings = get_total_crossings(TEST_DEGREE)
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    inv_gate = UnitaryGate(Operator(QuantumCircuit(8).compose(qft_exact.inverse()).compose(V_prep.inverse())).data, label="Inv")

    dd_ratios = np.linspace(0.0, 1.0, 5) 
    gate_errors_1q = np.arange(0.0, 0.026, 0.005) # 0.0% to 2.5% 1Q error
    
    fidelity_matrix = np.zeros((len(gate_errors_1q), len(dd_ratios)))

    for i, err in enumerate(gate_errors_1q):
        print(f"  Testing 1Q Error Rate: {err*100:.2f}%")
        for j, ratio in enumerate(dd_ratios):
            m3_fidelities = []
            for scale in [1, 3, 5]:
                circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
                circ.compose(V_prep, qubits=data_qubits, inplace=True)
                
                build_dd_sweep_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), 
                                  TEST_DEGREE, ratio, total_crossings, scale)
                                  
                circ.append(inv_gate, data_qubits); circ.measure(data_qubits, cr_out)

                # Isolate 1Q error: 2Q error=0.0, SPAM=0.0, T1/T2=inf
                model = get_comprehensive_noise_model(t1_us=np.inf, t2_us=np.inf, pulse_err_1q=err*scale, 
                                                     pulse_err_2q=0.0, spam_error=0.0)
                
                counts = run_local_simulation(circ, noise_model=model, shots=2000)
                m3_fidelities.append(apply_matrix_inversion(counts, 2000, 0.0))
            
            mitigated_fid = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8
            fidelity_matrix[i, j] = min(max(mitigated_fid, 0), 1.0)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(fidelity_matrix * 100, cmap='plasma', origin='lower', aspect='auto')
    plt.colorbar(label='Mitigated Fidelity (%)')
    plt.xticks(range(len(dd_ratios)), [f"{int(r*100)}%" for r in dd_ratios])
    plt.yticks(range(len(gate_errors_1q)), [f"{e*100:.1f}%" for e in gate_errors_1q])
    plt.xlabel('DD Ratio'); plt.ylabel('Physical 1Q Gate Error')
    plt.title(f'Degree {TEST_DEGREE}: 1Q Noise vs DD Ratio')
    plt.savefig(f'degree_{TEST_DEGREE}_1q_noise_vs_dd.png', dpi=300)
    print(f"Saved degree_{TEST_DEGREE}_1q_noise_vs_dd.png")

if __name__ == "__main__":
    for deg in range(8):
        execute_noise_vs_dd_sweep(TEST_DEGREE=deg)