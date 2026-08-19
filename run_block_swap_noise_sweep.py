import os
import warnings
import numpy as np
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_combined_threshold_boundaries(matrices_by_degree, x_values, y_values, x_label, y_label, 
                                       x_tick_format, y_tick_format, filename, title, folder_name, threshold=0.8):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap('tab10').colors
    
    legend_lines = []
    
    for deg, data in matrices_by_degree.items():
        color = colors[deg % len(colors)]
        rows, cols = data.shape
        legend_lines.append(mlines.Line2D([], [], color=color, lw=3, label=f'Degree {deg}'))
        
        for i in range(rows):
            for j in range(cols):
                # 檢查右側鄰居 (跨越 threshold)
                if j < cols - 1:
                    if (data[i, j] >= threshold) != (data[i, j+1] >= threshold):
                        ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, lw=3)
                # 檢查上方鄰居 (跨越 threshold)
                if i < rows - 1:
                    if (data[i, j] >= threshold) != (data[i+1, j] >= threshold):
                        ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, lw=3)

    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([x_tick_format(x) for x in x_values])
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([y_tick_format(y) for y in y_values])
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 畫上淺色背景網格，對齊實際數據點
    ax.set_xticks(np.arange(-0.5, len(x_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_values), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='--', linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    
    ax.legend(handles=legend_lines, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    plt.savefig(f'{folder_name}/{filename}.png', dpi=300)
    plt.close()

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator

# 假設你原本的 src 模組結構不變
from src.circuits import create_registers, build_state_preparation
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion

# ==========================================
# 1. Block Swap 專用的 DD Builder (終極修正版)
# ==========================================
def build_block_swap_qft_dd(circuit, data_qubits, approx_degree=4, dd_ratio=0.0, zne_scale=1, phase_drift_per_epoch=0.0):
    """
    使用 Block Swap (Pipelined) 架構的 QFT。
    """
    p2l = list(range(8))
    
    total_crossings = 4 
    num_dd_epochs = int(round(dd_ratio * total_crossings))
    protection_map = ['DD'] * num_dd_epochs + ['None'] * (total_crossings - num_dd_epochs)
    if len(protection_map) > 0:
        np.random.shuffle(protection_map)

    def cp_tracked(p_a, p_b):
        log_a = p2l[p_a]
        log_b = p2l[p_b]
        dist = abs(log_a - log_b)
        
        # 精準對齊 Qiskit 的 approximation_degree 邏輯
        # 如果 degree=0，保留所有 CP；如果 degree>0，只保留距離 <= degree 的 CP
        keep_gate = True
        max_allowed_dist = 7 - approx_degree
            
        if dist <= max_allowed_dist:
            angle = np.pi / (2 ** dist)
            circuit.cp(angle, data_qubits[p_a], data_qubits[p_b])

    def swap_tracked(p_a, p_b):
        # 🔥 致命錯誤修正：同上，確保 SWAP 打在真正的資料位元上
        circuit.swap(data_qubits[p_a], data_qubits[p_b])
        p2l[p_a], p2l[p_b] = p2l[p_b], p2l[p_a]

    # --- 階段 1: Local QFT on Block A ---
    for i in range(4):
        circuit.h(data_qubits[i])
        for j in range(i + 1, 4):
            cp_tracked(j, i)
            
    circuit.barrier()

    # --- 階段 2: Block Swap 交會與 DD 注入 ---
    for k in range(4):
        a = k
        b = 7 - k
        
        cp_tracked(a, b)
        swap_tracked(a, b)
        
        for m in range(4, b):
            cp_tracked(b, m)
        for n in range(a + 1, 4):
            cp_tracked(a, n)
            
        # DD 注入邏輯 (這裡之前有乖乖用 data_qubits，所以沒事)
        if len(protection_map) > 0:
            strat = protection_map.pop(0)
            epoch_depth_factor = (total_crossings - len(protection_map))
            variable_drift = phase_drift_per_epoch * epoch_depth_factor * zne_scale
            
            idle_indices = [idx for idx in range(8) if idx not in (a, b)]
            for idx in idle_indices:
                phys = data_qubits[idx]
                if strat == 'DD':
                    circuit.rz(variable_drift / 2, phys)
                    circuit.x(phys)
                    circuit.rz(variable_drift / 2, phys)
                    circuit.x(phys)
                else:
                    circuit.rz(variable_drift, phys)
                    
    circuit.barrier()

    # --- 階段 3: Local QFT on Block B ---
    for idx_i in range(4):
        phys_i = 3 - idx_i
        circuit.h(data_qubits[phys_i])
        for idx_j in range(idx_i + 1, 4):
            phys_j = 3 - idx_j
            cp_tracked(phys_j, phys_i)
            
    circuit.barrier()
    return p2l

# ==========================================
# 2. 獨立的核心評估函數 (ZNE 流程 - 修正版)
# ==========================================
def get_zne_mitigated_fidelity_block_swap(test_degree, shots, builder_kwargs, t1_us, t2_us, err_1q, err_2q, spam_err):
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=True).reverse_bits()
    
    ideal_inverse = QuantumCircuit(8)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    inv_gate = UnitaryGate(Operator(ideal_inverse).data, label="Ideal_Uncompute")
    
    m3_fidelities = []
    # 執行 ZNE 縮放
    for scale in [1, 3, 5]:
        circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
        circ.compose(V_prep, qubits=data_qubits, inplace=True)
        circ.barrier()
        
        # 執行 Block Swap，取得 Tracker，但我們不拿它來綁定測量了
        build_block_swap_qft_dd(
            circ, data_qubits, approx_degree=test_degree, zne_scale=scale, **builder_kwargs
        )
        
        circ.barrier()
        circ.append(inv_gate, data_qubits)
        circ.measure(data_qubits, cr_out)

        # 設定雜訊模型
        model = get_comprehensive_noise_model(
            t1_us=t1_us/scale if not np.isinf(t1_us) else np.inf, 
            t2_us=t2_us/scale if not np.isinf(t2_us) else np.inf, 
            pulse_err_1q=err_1q*scale, 
            pulse_err_2q=err_2q*scale, 
            spam_error=spam_err*scale
        )
        
        counts = run_local_simulation(circ, noise_model=model, shots=shots)
        m3_fid = apply_matrix_inversion(counts, shots, p_error=spam_err if spam_err == 0.0 else 0.02)
        m3_fidelities.append(m3_fid)
    
    # Richardson Extrapolation
    mitigated_fid = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8
    return min(max(mitigated_fid, 0), 1.0)


# ==========================================
# 3. 實驗排程與繪圖架構 (繼承自原本的批次框架)
# ==========================================
LOCAL_TEST_MODE = False
DEGREES_TO_TEST = [1] if LOCAL_TEST_MODE else list(range(8))
STEPS = 3 if LOCAL_TEST_MODE else 11
SHOTS = 100 if LOCAL_TEST_MODE else 3000

os.makedirs("results_blockswap", exist_ok=True)

def produce_2d_heatmap(target_degree, x_values, y_values, x_label, y_label, 
                       x_tick_format, y_tick_format, config_mapper, filename, title):
    print(f"[{title}] - Degree {target_degree}")
    fidelity_matrix = np.zeros((len(y_values), len(x_values)))
    
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            experiment_kwargs = config_mapper(x_val, y_val)
            fid = get_zne_mitigated_fidelity_block_swap(test_degree=target_degree, shots=SHOTS, **experiment_kwargs)
            fidelity_matrix[i, j] = fid
            print(f"  └─ X: {x_tick_format(x_val):>5} | Y: {y_tick_format(y_val):>7} -> Fidelity: {fid*100:5.1f}%")

    plt.figure(figsize=(10, 8))
    dx = (x_values[1] - x_values[0]) / 2 if len(x_values) > 1 else 0.5
    dy = (y_values[1] - y_values[0]) / 2 if len(y_values) > 1 else 0.5
    extent = [x_values[0]-dx, x_values[-1]+dx, y_values[0]-dy, y_values[-1]+dy]
    
    im = plt.imshow(fidelity_matrix * 100, cmap='plasma', aspect='auto', origin='lower', extent=extent, vmin=0, vmax=100)
    cbar = plt.colorbar(im)
    cbar.set_label('Mitigated Output Fidelity (%)', rotation=270, labelpad=20)
    
    plt.xticks(x_values, [x_tick_format(x) for x in x_values])
    plt.yticks(y_values, [y_tick_format(y) for y in y_values])
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'{title} (Degree {target_degree})', fontsize=14)
    
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            val = fidelity_matrix[i, j] * 100
            text_color = "black" if val > 70 else "white"
            plt.text(x_values[j], y_values[i], f'{val:.1f}', ha="center", va="center", color=text_color, fontsize=9)
                     
    plt.tight_layout()
    plt.savefig(f'results_blockswap/{filename}.png', dpi=300)

    np.save(f'results_blockswap/{filename}.npy', fidelity_matrix)
    
    plt.close()

    return fidelity_matrix

# ==========================================
# 定義 4 個實驗的 Mapping (與原本相同)[cite: 5]
# ==========================================
def run_all_experiments():
    x_vals = np.linspace(0.0, 1.0, STEPS)
    y_vals_1q = np.linspace(0.0, 0.01, STEPS)
    y_vals_2q = np.linspace(0.0, 0.025, STEPS)
    y_vals_spam = np.linspace(0.0, 0.025, STEPS)
    y_vals_dec = np.linspace(0.0, 0.05, STEPS)
    
    mat_1q, mat_2q, mat_spam, mat_dec = {}, {}, {}, {}
    
    for deg in DEGREES_TO_TEST:
        # Exp 1: 1Q Error
        mat_1q[deg] = produce_2d_heatmap(
            deg, x_vals, y_vals_1q, "DD Ratio", "1Q Gate Error (%)",
            lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%",
            lambda x, y: {'builder_kwargs': {'dd_ratio': x, 'phase_drift_per_epoch': 0.0}, 't1_us': np.inf, 't2_us': np.inf, 'err_1q': y, 'err_2q': 0.0, 'spam_err': 0.0},
            f"Exp1_BS_1Q_vs_DD_deg{deg}", "Block Swap: 1Q Error vs DD"
        )
        
        # Exp 2: 2Q Error
        mat_2q[deg] = produce_2d_heatmap(
            deg, x_vals, y_vals_2q, "DD Ratio", "2Q Gate Error (%)",
            lambda x: f"{x:.1f}", lambda y: f"{y*100:.2f}%",
            lambda x, y: {'builder_kwargs': {'dd_ratio': x, 'phase_drift_per_epoch': 0.0}, 't1_us': np.inf, 't2_us': np.inf, 'err_1q': 0.0, 'err_2q': y, 'spam_err': 0.0},
            f"Exp2_BS_2Q_vs_DD_deg{deg}", "Block Swap: 2Q Error vs DD"
        )
        
        # Exp 3: SPAM Error
        mat_spam[deg] = produce_2d_heatmap(
            deg, x_vals, y_vals_spam, "DD Ratio", "SPAM Error (%)",
            lambda x: f"{x:.1f}", lambda y: f"{y*100:.2f}%",
            lambda x, y: {'builder_kwargs': {'dd_ratio': x, 'phase_drift_per_epoch': 0.0}, 't1_us': np.inf, 't2_us': np.inf, 'err_1q': 0.0, 'err_2q': 0.0, 'spam_err': y},
            f"Exp3_BS_SPAM_vs_DD_deg{deg}", "Block Swap: SPAM vs DD"
        )
        
        # Exp 4: Decoherence
        mat_dec[deg] = produce_2d_heatmap(
            deg, x_vals, y_vals_dec, "DD Ratio", r"Decoherence Rate $\Gamma = 1/T$ ($1/\mu s$)",
            lambda x: f"{x:.1f}", lambda y: f"inf" if y==0 else f"{int(1.0/y)}us",
            lambda x, y: {'builder_kwargs': {'dd_ratio': x, 'phase_drift_per_epoch': (np.pi / 48) * (y / 0.02) if y > 0 else 0.0}, 't1_us': 1.0/y if y>0 else np.inf, 't2_us': 1.0/y if y>0 else np.inf, 'err_1q': 0.0, 'err_2q': 0.0, 'spam_err': 0.0},
            f"Exp4_BS_Decoherence_vs_DD_deg{deg}", "Block Swap: T1/T2 vs DD"
        )

    # 迴圈結束後，統一畫出 4 張 80% Threshold 的整合圖
    print("\n繪製整合邊界圖中...")
    folder = "results_blockswap"
    plot_combined_threshold_boundaries(mat_1q, x_vals, y_vals_1q, "DD Ratio", "1Q Gate Error (%)", lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%", "Exp1_BS_Combined", "Block Swap: 1Q Error (80% Threshold)", folder)
    plot_combined_threshold_boundaries(mat_2q, x_vals, y_vals_2q, "DD Ratio", "2Q Gate Error (%)", lambda x: f"{x:.1f}", lambda y: f"{y*100:.2f}%", "Exp2_BS_Combined", "Block Swap: 2Q Error (80% Threshold)", folder)
    plot_combined_threshold_boundaries(mat_spam, x_vals, y_vals_spam, "DD Ratio", "SPAM Error (%)", lambda x: f"{x:.1f}", lambda y: f"{y*100:.2f}%", "Exp3_BS_Combined", "Block Swap: SPAM Error (80% Threshold)", folder)
    plot_combined_threshold_boundaries(mat_dec, x_vals, y_vals_dec, "DD Ratio", r"Decoherence Rate $\Gamma = 1/T$ ($1/\mu s$)", lambda x: f"{x:.1f}", lambda y: f"inf" if y==0 else f"{int(1.0/y)}us", "Exp4_BS_Combined", "Block Swap: T1/T2 (80% Threshold)", folder)

if __name__ == "__main__":
    print("====== 啟動 Block Swap 架構雜訊對比實驗 ======\n")
    run_all_experiments()
    print("\n====== 所有實驗執行完畢！結果已存入 results_blockswap 目錄 ======")