import numpy as np
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
def execute_hybrid_pipeline():
    print("--- Hybrid QEM Pipeline: Comprehensive Noise (M3 + ZNE) ---")
    
    num_shots = 10000
    scales = [1, 3, 5]
    raw_fidelities = []
    m3_fidelities = []
    
    # 1. Base Realistic Hardware Parameters (IBM Marrakesh approximations)
    base_err_1q = 0.001
    base_err_2q = 0.01
    base_spam = 0.02

    
    circ = generate_base_circuit(approx_degree=0)
    
    # 2. Iterate across the noise scaling domains
    for scale in scales:
        print(f"\nExecuting Noise Scale λ = {scale}...")
        
        # FIX: Removed the trailing commas so these are floats, not tuples.
        scaled_t1 = np.inf
        scaled_t2 = np.inf
        
        scaled_1q = base_err_1q * scale
        scaled_2q = base_err_2q * scale
        scaled_spam = base_spam * scale
        
        # T1 and T2 are now mathematically infinite (turned off)
        scaled_model = get_comprehensive_noise_model(
            t1_us=scaled_t1, t2_us=scaled_t2, 
            pulse_err_1q=scaled_1q, pulse_err_2q=scaled_2q, 
            spam_error=scaled_spam
        )
        
        counts = run_local_simulation(circ, noise_model=scaled_model, shots=num_shots)
        
        # Capture Raw Fidelity
        raw_fid = counts.get('00000000', 0) / num_shots
        raw_fidelities.append(raw_fid)
        
        # Step 1 of Hybrid QEM: Apply Matrix Inversion to this specific scale's counts
        m3_fid = apply_matrix_inversion(counts, num_shots, p_error=scaled_spam)
        m3_fidelities.append(m3_fid)
        
        print(f" -> Raw Hardware Fidelity: {raw_fid * 100:.2f}%")
        print(f" -> M3 Mitigated Fidelity: {m3_fid * 100:.2f}%")

    # Step 2 of Hybrid QEM: Apply ZNE (Richardson Extrapolation) to the M3-cleaned data
    f1, f3, f5 = m3_fidelities
    hybrid_mitigated_fid = (15 * f1 - 10 * f3 + 3 * f5) / 8
    
    print("\n" + "="*50)
    print(" HYBRID QEM RESULTS (M3 + ZNE) [T1/T2 ISOLATED]")
    print("="*50)
    print(f"Baseline Hardware Fidelity (λ=1):      {raw_fidelities[0] * 100:.2f}%")
    print("-" * 50)
    print(f"M3-Only Mitigated Fidelity (λ=1):      {m3_fidelities[0] * 100:.2f}%")
    print(f"Fully Mitigated Fidelity (M3 + ZNE):   {min(max(hybrid_mitigated_fid * 100, 0), 100.0):.2f}%")
    print("="*50)

if __name__ == "__main__":
    execute_hybrid_pipeline()