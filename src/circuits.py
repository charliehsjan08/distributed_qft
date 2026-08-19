import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import IGate

def create_registers():
    """Initializes the data and communication registers for the distributed architecture."""
    qr_data_A = QuantumRegister(4, 'data_a')
    qr_comm_A = QuantumRegister(1, 'comm_a')
    qr_data_B = QuantumRegister(4, 'data_b')
    qr_comm_B = QuantumRegister(1, 'comm_b')
    cr_tele = ClassicalRegister(2, 'c_tele')
    cr_out = ClassicalRegister(8, 'c_out')
    return qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out

def build_state_preparation(data_qubits):
    """Generates an ideal randomized initial state vector across the data registers."""
    V_prep_logical = random_circuit(8, depth=3, measure=False, seed=42)
    return transpile(V_prep_logical, basis_gates=['cx', 'rz', 'sx', 'x'])

def build_distributed_qft(circuit, registers, err_1q=None, err_2q=None,approx_degree=0):
    """
    Constructs the Stage-2 Distributed QFT logic.
    If err_1q and err_2q are provided, synthetic noise is injected inline.
    If None, a pristine hardware-targeted circuit is built.
    """
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, _ = registers
    
    def get_physical_qubit(logical_index):
        return qr_data_A[logical_index] if logical_index < 4 else qr_data_B[logical_index - 4]

    qft_perfect = QFT(num_qubits=8, approximation_degree=approx_degree, do_swaps=False)
    unrolled_perfect = transpile(qft_perfect, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

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

                circuit.reset(qr_comm_A)
                circuit.reset(qr_comm_B)
                circuit.h(qr_comm_A)
                if err_1q: circuit.append(err_1q, [qr_comm_A])
                circuit.cx(qr_comm_A, qr_comm_B)
                if err_2q: circuit.append(err_2q, [qr_comm_A, qr_comm_B])

                if is_ctrl_on_A:
                    circuit.cx(phys_ctrl, qr_comm_A)
                    if err_2q: circuit.append(err_2q, [phys_ctrl, qr_comm_A])
                    circuit.cx(qr_comm_B, phys_targ)
                    if err_2q: circuit.append(err_2q, [qr_comm_B, phys_targ])

                    circuit.h(qr_comm_B)
                    if err_1q: circuit.append(err_1q, [qr_comm_B])
                    circuit.measure(qr_comm_A, cr_tele[0])
                    circuit.measure(qr_comm_B, cr_tele[1])

                    with circuit.if_test((cr_tele[1], 1)):
                        circuit.z(phys_ctrl)
                        if err_1q: circuit.append(err_1q, [phys_ctrl])
                    with circuit.if_test((cr_tele[0], 1)):
                        circuit.x(phys_targ)
                        if err_1q: circuit.append(err_1q, [phys_targ])
                else:
                    circuit.cx(phys_ctrl, qr_comm_B)
                    if err_2q: circuit.append(err_2q, [phys_ctrl, qr_comm_B])
                    circuit.cx(qr_comm_A, phys_targ)
                    if err_2q: circuit.append(err_2q, [qr_comm_A, phys_targ])

                    circuit.h(qr_comm_A)
                    if err_1q: circuit.append(err_1q, [qr_comm_A])
                    circuit.measure(qr_comm_B, cr_tele[1])
                    circuit.measure(qr_comm_A, cr_tele[0])

                    with circuit.if_test((cr_tele[0], 1)):
                        circuit.z(phys_ctrl)
                        if err_1q: circuit.append(err_1q, [phys_ctrl])
                    with circuit.if_test((cr_tele[1], 1)):
                        circuit.x(phys_targ)
                        if err_1q: circuit.append(err_1q, [phys_targ])
            else:
                phys_args = [get_physical_qubit(q) for q in logical_qubits]
                circuit.cx(*phys_args)
                if err_2q: circuit.append(err_2q, phys_args)
        else:
            phys_args = [get_physical_qubit(q) for q in logical_qubits]
            circuit.append(inst.operation, phys_args)
            if err_1q: circuit.append(IGate(), phys_args)   
                     
    return unrolled_perfect