import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from src.circuits import create_registers, build_state_preparation, build_distributed_qft
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
from qiskit.circuit.library import UnitaryGate

def generate_base_circuit(approx_degree):
    """
    Generates the QFT architecture with an APPROXIMATE forward path 
    and a mathematically PERFECT, NOISELESS inverse path.
    """
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
    circ.compose(V_prep, qubits=data_qubits, inplace=True)
    circ.barrier()
    
    # 1. Forward Path (Noisy, Approximate, Hardware-Dependent)
    _ = build_distributed_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), approx_degree=approx_degree)
    circ.barrier()
    
    # 2. Conceptual Exact Inverse (Not added to circuit yet)
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    ideal_inverse = QuantumCircuit(8)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    
    # 3. The Noiseless Bypass
    # Convert the inverse to a pure matrix, and wrap it in a UnitaryGate
    inv_matrix = Operator(ideal_inverse).data
    inv_gate = UnitaryGate(inv_matrix, label="Ideal_Uncompute")
    
    # Append the pure math gate. Aer will simulate this instantly with 0% noise.
    circ.append(inv_gate, data_qubits)
    circ.barrier()
    
    # 4. Final Measurement (Subject to SPAM, fixed by M3)
    circ.measure(data_qubits, cr_out)
    
    return circ

def run_spam_experiments():
    num_shots = 10000
    circ = generate_base_circuit(approx_degree=0)
    
    print("\n--- TEST 1: Matrix Inversion Impact (Fixed 1% Error) ---")
    fixed_p = 0.01
    # ISOLATION: Turn OFF T1/T2 (np.inf) and Gate Errors (0), leaving ONLY SPAM
    isolated_spam_model = get_comprehensive_noise_model(
        t1_us=np.inf, t2_us=np.inf, pulse_err_1q=0.0, pulse_err_2q=0.0, spam_error=fixed_p
    )
    counts_fixed = run_local_simulation(circ, noise_model=isolated_spam_model, shots=num_shots)
    
    raw_fid_fixed = counts_fixed.get('00000000', 0) / num_shots
    mitigated_fid_fixed = apply_matrix_inversion(counts_fixed, num_shots, fixed_p)
    
    print(f"Raw Hardware Fidelity:   {raw_fid_fixed * 100:.2f}%")
    print(f"Mitigated Fidelity:      {mitigated_fid_fixed * 100:.2f}%")
    print("-" * 55)
    
    print("\n--- TEST 2: Varying SPAM Error (0% to 15%) ---")
    error_rates = np.linspace(0.0, 0.15, 10)
    raw_fidelities = []
    mitigated_fidelities = []
    
    for p in error_rates:
        print(f"Simulating SPAM error p = {p:.3f}...")
        # ISOLATION: Dynamically sweep 'p' while keeping everything else off
        sweep_model = get_comprehensive_noise_model(
            t1_us=np.inf, t2_us=np.inf, pulse_err_1q=0.0, pulse_err_2q=0.0, spam_error=p
        )
        temp_counts = run_local_simulation(circ, noise_model=sweep_model, shots=num_shots)
        
        raw_f = temp_counts.get('00000000', 0) / num_shots
        mit_f = apply_matrix_inversion(temp_counts, num_shots, p)
        
        raw_fidelities.append(raw_f)
        mitigated_fidelities.append(mit_f)

    # Diagram Generation
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates * 100, [f * 100 for f in raw_fidelities], marker='o', linestyle='-', color='red', label='Raw Fidelity (Unmitigated)')
    plt.plot(error_rates * 100, [f * 100 for f in mitigated_fidelities], marker='s', linestyle='--', color='blue', label='Mitigated Fidelity (Matrix Inversion)')
    
    plt.title('Impact of SPAM Error Mitigation on 8-Qubit QFT')
    plt.xlabel('Physical Readout Error Rate (%)')
    plt.ylabel('Algorithm Fidelity (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('spam(only last 8 readout)_mitigation_results.png', dpi=300)
    print("\nExperiment complete! Diagram saved as 'spam_mitigation_results.png'.")

if __name__ == "__main__":
    run_spam_experiments()