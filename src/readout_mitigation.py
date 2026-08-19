def apply_matrix_inversion(raw_counts, num_shots, p_error, target_state='00000000'):
    """
    Applies Tensored Readout Mitigation (Matrix Inversion) mathematically.
    Uses Hamming distance to avoid building a 256x256 inversion matrix.
    """
    mitigated_target_prob = 0.0
    num_qubits = len(target_state)
    
    normalization_factor = 1.0 / ((1 - 2 * p_error) ** num_qubits)

    for bitstring, count in raw_counts.items():
        k = sum(1 for a, b in zip(bitstring, target_state) if a != b)
        weight = normalization_factor * ((1 - p_error) ** (num_qubits - k)) * ((-p_error) ** k)
        mitigated_target_prob += (count / num_shots) * weight

    return max(0.0, min(mitigated_target_prob, 1.0))