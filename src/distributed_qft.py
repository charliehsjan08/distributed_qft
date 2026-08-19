import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

class DistributedQFTBuilder:
    def __init__(self, circuit, registers, dd_ratio=0.0, zne_scale=1, phase_drift=0.0):
        self.qc = circuit
        self.qr_data_A, self.qr_comm_A, self.qr_comm_B, self.qr_data_B, self.cr_tele, _ = registers
        self.dd_ratio = dd_ratio
        self.zne_scale = zne_scale
        self.phase_drift = phase_drift
        self.p2l = list(range(8)) # Physical to Logical
        self.total_epochs = 0
        self.dd_count = 0

    def get_phys(self, p_idx):
        return self.qr_data_A[p_idx] if p_idx < 4 else self.qr_data_B[p_idx - 4]

    def is_cross_chip(self, p_a, p_b):
        return (p_a < 4) != (p_b < 4)

    def apply_dd(self, active_p):
        """Apply DD pulses to all qubits except those active in the current gate."""
        if self.dd_count >= int(round(self.dd_ratio * 64)): # 估計總Epochs約64
            return
        
        # 決定這次是否執行 DD (簡化版: 依比例隨機或順序)
        if np.random.random() < self.dd_ratio:
            self.dd_count += 1
            drift = self.phase_drift * self.zne_scale
            for p in range(8):
                if p not in active_p:
                    phys = self.get_phys(p)
                    self.qc.rz(drift / 2, phys)
                    self.qc.x(phys)
                    self.qc.rz(drift / 2, phys)
                    self.qc.x(phys)

    def telegate_cx(self, p_ctrl, p_targ):
        """Standard Telegate Bridge for CNOT"""
        self.total_epochs += 1
        phys_ctrl = self.get_phys(p_ctrl)
        phys_targ = self.get_phys(p_targ)
        
        self.qc.reset(self.qr_comm_A)
        self.qc.reset(self.qr_comm_B)
        self.qc.h(self.qr_comm_A)
        self.qc.cx(self.qr_comm_A, self.qr_comm_B)
        
        if p_ctrl < 4: # Ctrl on A, Targ on B
            self.qc.cx(phys_ctrl, self.qr_comm_A)
            self.qc.cx(self.qr_comm_B, phys_targ)
            self.qc.h(self.qr_comm_B)
            self.qc.measure(self.qr_comm_A, self.cr_tele[0])
            self.qc.measure(self.qr_comm_B, self.cr_tele[1])
            with self.qc.if_test((self.cr_tele[1], 1)): self.qc.z(phys_ctrl)
            with self.qc.if_test((self.cr_tele[0], 1)): self.qc.x(phys_targ)
        else: # Ctrl on B, Targ on A
            self.qc.cx(phys_ctrl, self.qr_comm_B)
            self.qc.cx(self.qr_comm_A, phys_targ)
            self.qc.h(self.qr_comm_A)
            self.qc.measure(self.qr_comm_B, self.cr_tele[1])
            self.qc.measure(self.qr_comm_A, self.cr_tele[0])
            with self.qc.if_test((self.cr_tele[0], 1)): self.qc.z(phys_ctrl)
            with self.qc.if_test((self.cr_tele[1], 1)): self.qc.x(phys_targ)
        
        self.apply_dd([p_ctrl, p_targ])

    def cp_tracked(self, p_a, p_b):
        log_a, log_b = self.p2l[p_a], self.p2l[p_b]
        angle = np.pi / (2 ** abs(log_a - log_b))
        
        if not self.is_cross_chip(p_a, p_b):
            self.qc.cp(angle, self.get_phys(p_a), self.get_phys(p_b))
        else:
            # CP via CNOT decomposition: CP(theta) = Rz(theta/2) + CNOT + Rz(-theta/2) + CNOT
            # 簡化實作：使用 Telegate-CNOT 實現跨晶片 CP
            self.qc.rz(angle/2, self.get_phys(p_a))
            self.telegate_cx(p_a, p_b)
            self.qc.rz(-angle/2, self.get_phys(p_a))
            self.telegate_cx(p_a, p_b)

    def swap_tracked(self, p_a, p_b):
        if not self.is_cross_chip(p_a, p_b):
            self.qc.swap(self.get_phys(p_a), self.get_phys(p_b))
        else:
            # Cross-chip SWAP via 3 CNOTs
            self.telegate_cx(p_a, p_b)
            self.telegate_cx(p_b, p_a)
            self.telegate_cx(p_a, p_b)
        self.p2l[p_a], self.p2l[p_b] = self.p2l[p_b], self.p2l[p_a]

    def build(self):
        # Phase 1: Local A
        for i in range(4):
            self.qc.h(self.get_phys(i))
            for j in range(i+1, 4):
                self.cp_tracked(j, i)
        
        self.qc.barrier(label="Symmetric Bridge")

        # Phase 2: Symmetric Bridge
        for k in range(4):
            a, b = k, 7 - k
            self.cp_tracked(a, b)
            self.swap_tracked(a, b)
            for m in range(4, b): self.cp_tracked(b, m)
            for n in range(a+1, 4): self.cp_tracked(a, n)

        self.qc.barrier(label="Local B")

        # Phase 3: Local B
        for idx_i in range(4):
            phys_i = 3 - idx_i
            self.qc.h(self.get_phys(phys_i))
            for idx_j in range(idx_i + 1, 4):
                phys_j = 3 - idx_j
                self.cp_tracked(phys_j, phys_i)
        return self.p2l