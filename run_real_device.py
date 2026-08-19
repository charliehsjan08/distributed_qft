import os
import warnings
import numpy as np
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_gate_map
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# 引入你的 DD Builder (Pure Telegate 會用到)
from qft_experiment_harness import build_test_circuit_dd
from src.circuits import build_state_preparation


# ==========================================
# 1. 硬體執行與 6-Qubit 指定設定區
# ==========================================
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="DkFK6rbyiPxXXMGhEgQKUNJaDMQyK8OYUT8goZsQdATj")
BACKEND_NAME = "ibm_brisbane" 
backend = service.backend(BACKEND_NAME)

# 🔥 核心設定：指定 6 個 Qubit 位置
# 邏輯順序：[DA0, DA1, CA, CB, DB0, DB1]
# 請挑選一組在 IBM 晶片上盡可能緊湊的 6 顆 Qubits
PHYSICAL_QUBITS = [0, 1, 2, 3, 14, 15] 

SHOTS = 4000
# 4-qubit QFT 的最大 degree 就是 3 (0~3)
DEGREES = [0, 1, 2, 3]       
DD_STEPS = 11                  
DD_RATIOS = np.linspace(0.0, 1.0, DD_STEPS) 

os.makedirs("results_ibm_4q", exist_ok=True)

# ==========================================
# 2. 覆寫 4-qubit 專用的 Register 建立
# ==========================================
def create_4q_registers():
    qr_data_A = QuantumRegister(2, 'data_A')
    qr_comm_A = QuantumRegister(1, 'comm_A')
    qr_comm_B = QuantumRegister(1, 'comm_B')
    qr_data_B = QuantumRegister(2, 'data_B')
    cr_tele = ClassicalRegister(2, 'tele_meas') # 若你的 Pure Telegate 需要用到
    cr_out = ClassicalRegister(4, 'out_meas')
    return qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out

# ==========================================
# 3. 4-Qubit 專用的 Block Swap DD Builder
# ==========================================
def build_4q_block_swap_qft_dd(circuit, data_qubits, approx_degree=2, dd_ratio=0.0):
    p2l = list(range(4))
    total_crossings = 2 
    num_dd_epochs = int(round(dd_ratio * total_crossings))
    protection_map = ['DD'] * num_dd_epochs + ['None'] * (total_crossings - num_dd_epochs)
    if len(protection_map) > 0:
        np.random.shuffle(protection_map)

    def cp_tracked(p_a, p_b):
        dist = abs(p2l[p_a] - p2l[p_b])
        # 4-qubit 中最大距離是 3。限制條件等比例縮放
        if dist <= 3 - approx_degree:
            circuit.cp(np.pi / (2 ** dist), data_qubits[p_a], data_qubits[p_b])

    def swap_tracked(p_a, p_b):
        circuit.swap(data_qubits[p_a], data_qubits[p_b])
        p2l[p_a], p2l[p_b] = p2l[p_b], p2l[p_a]

    # --- Block A (2 qubits) ---
    for i in range(2):
        circuit.h(data_qubits[i])
        for j in range(i + 1, 2):
            cp_tracked(j, i)
    circuit.barrier()

    # --- 交會與 DD ---
    for k in range(2):
        a, b = k, 3 - k
        cp_tracked(a, b)
        swap_tracked(a, b)
        
        for m in range(2, b): cp_tracked(b, m)
        for n in range(a + 1, 2): cp_tracked(a, n)
            
        if len(protection_map) > 0:
            strat = protection_map.pop(0)
            idle_indices = [idx for idx in range(4) if idx not in (a, b)]
            for idx in idle_indices:
                phys = data_qubits[idx]
                if strat == 'DD':
                    # 真機 Echo
                    circuit.x(phys)
                    circuit.x(phys)
    circuit.barrier()

    # --- Block B (2 qubits) ---
    for idx_i in range(2):
        phys_i = 1 - idx_i
        circuit.h(data_qubits[phys_i])
        for idx_j in range(idx_i + 1, 2):
            phys_j = 1 - idx_j
            cp_tracked(phys_j, phys_i)
    circuit.barrier()
    return p2l

# ==========================================
# 4. 建立單一電路 (包含 Ideal Uncompute)
# ==========================================
def create_4q_uncompute_circuit(architecture, degree, dd_ratio):
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_4q_registers()
    # Data qubits 共 4 顆
    data_qubits = [qr_data_A[0], qr_data_A[1], qr_data_B[0], qr_data_B[1]]
    V_prep = build_state_preparation(data_qubits)
    
    qft_exact = QFT(num_qubits=4, approximation_degree=0, do_swaps=True).reverse_bits()
    ideal_inverse = QuantumCircuit(4)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    inv_gate = UnitaryGate(Operator(ideal_inverse).data, label="Ideal_Uncompute")
    
    # 6 顆 Qubit 全數加入線路
    circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
    circ.compose(V_prep, qubits=data_qubits, inplace=True)
    circ.barrier()
    
    if architecture == "Pure Telegate":
        # 假設你的 Pure Telegate 也能動態適應 data_qubits 長度
        build_test_circuit_dd(circ, data_qubits, approx_degree=degree, dd_ratio=dd_ratio, phase_drift_per_epoch=0.0)
    else:
        build_4q_block_swap_qft_dd(circ, data_qubits, approx_degree=degree, dd_ratio=dd_ratio)
        
    circ.barrier()
    circ.append(inv_gate, data_qubits)
    circ.measure(data_qubits, cr_out) 
    
    return circ

# ==========================================
# 5. 繪圖與硬體派發
# ==========================================
def verify_and_plot_topology():
    print("🗺️ 正在繪製並驗證硬體拓樸結構...")
    fig = plot_gate_map(backend, highlight_qubits=PHYSICAL_QUBITS, plot_directed=False, label_qubits=True)
    fig.savefig("results_ibm_4q/hardware_topology_map.png", dpi=300)
    print("✅ 拓樸結構圖已儲存至 results_ibm_4q/hardware_topology_map.png")

def plot_2d_sweep(matrix, arch_name):
    plt.figure(figsize=(10, 8))
    extent = [DD_RATIOS[0], DD_RATIOS[-1], DEGREES[0], DEGREES[-1]]
    
    im = plt.imshow(matrix * 100, cmap='plasma', aspect='auto', origin='lower', extent=extent, vmin=0, vmax=100)
    cbar = plt.colorbar(im)
    cbar.set_label('Hardware Output Fidelity (%)', rotation=270, labelpad=20)
    
    plt.xticks(DD_RATIOS, [f"{x:.1f}" for x in DD_RATIOS])
    plt.yticks(DEGREES, [str(d) for d in DEGREES])
    plt.xlabel("DD Ratio", fontsize=12)
    plt.ylabel("AQFT Degree", fontsize=12)
    plt.title(f'{arch_name} (4-Qubit): Hardware Sweep', fontsize=14)
    
    for i in range(len(DEGREES)):
        for j in range(len(DD_RATIOS)):
            val = matrix[i, j] * 100
            text_color = "black" if val > 70 else "white"
            plt.text(DD_RATIOS[j], DEGREES[i], f'{val:.1f}', ha="center", va="center", color=text_color, fontsize=9)
                     
    plt.tight_layout()
    filename = arch_name.replace(" ", "")
    plt.savefig(f'results_ibm_4q/{filename}_4q_sweep.png', dpi=300)
    plt.close()

def run_hardware_sweep():
    print(f"📡 準備連接 Backend: {BACKEND_NAME}")
    verify_and_plot_topology()
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=PHYSICAL_QUBITS)
    
    circuits_to_run = []
    metadata_list = []
    
    print("⏳ 正在產生並 Transpile 量子線路...")
    for arch in ["Pure Telegate", "Block Swap"]:
        for deg in DEGREES:
            for dd in DD_RATIOS:
                circ = create_4q_uncompute_circuit(arch, deg, dd)
                isa_circ = pm.run(circ)
                circuits_to_run.append(isa_circ)
                metadata_list.append({"arch": arch, "degree": deg, "dd_ratio": dd})
                
    print("🚀 線路準備完畢，發送 Job 至 IBM Quantum...")
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    sampler.options.resilience_level = 1 
    
    job = sampler.run(circuits_to_run)
    job_id = job.job_id()
    print(f"✅ Job 已成功提交！Job ID: {job_id}")
    
    # === 若要等待結果並繪圖，請解除下方註解 ===
    # result = job.result()
    # print("🎉 運算完成！開始處理結果...")
    # mat_pure = np.zeros((len(DEGREES), len(DD_RATIOS)))
    # mat_bs = np.zeros((len(DEGREES), len(DD_RATIOS)))
    # 
    # for idx, pub_result in enumerate(result):
    #     meta = metadata_list[idx]
    #     counts = pub_result.data.out_meas.get_counts()
    #     success_shots = counts.get('0000', 0) # 4 qubit 全零
    #     fidelity = success_shots / SHOTS
    #     
    #     r_idx = DEGREES.index(meta["degree"])
    #     c_idx = list(DD_RATIOS).index(meta["dd_ratio"])
    #     
    #     if meta["arch"] == "Pure Telegate":
    #         mat_pure[r_idx, c_idx] = fidelity
    #     else:
    #         mat_bs[r_idx, c_idx] = fidelity
    #
    # np.save("results_ibm_4q/mat_pure_telegate.npy", mat_pure)
    # np.save("results_ibm_4q/mat_block_swap.npy", mat_bs)
    # plot_2d_sweep(mat_pure, "Pure Telegate")
    # plot_2d_sweep(mat_bs, "Block Swap")

if __name__ == "__main__":
    run_hardware_sweep()