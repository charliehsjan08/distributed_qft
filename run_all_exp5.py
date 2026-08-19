import os
import warnings
import numpy as np
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Operator

# 引入原有的模組
from qft_experiment_harness import get_zne_mitigated_fidelity, build_test_circuit_dd
from src.circuits import create_registers, build_state_preparation
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion

# ==========================================
# 執行環境設定
# ==========================================
LOCAL_TEST_MODE = False

if LOCAL_TEST_MODE:
    print("⚠️ 運行在 LOCAL TEST MODE (快速驗證)")
    DEGREES_TO_TEST = [4]
    STEPS = 3
    SHOTS = 100
else:
    print("🚀 運行在 FULL CLUSTER MODE")
    DEGREES_TO_TEST = list(range(8)) # 0 到 7
    STEPS = 11                       # 11 點網格
    SHOTS = 3000

# 建立統一輸出目錄
OUTPUT_DIR = "results_combined_noise"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Block Swap 專用的 DD Builder 與評估函數
# ==========================================
def build_block_swap_qft_dd(circuit, data_qubits, approx_degree=4, dd_ratio=0.0, zne_scale=1, phase_drift_per_epoch=0.0):
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
        
        keep_gate = True
        max_allowed_dist = 7 - approx_degree
            
        if dist <= max_allowed_dist:
            angle = np.pi / (2 ** dist)
            circuit.cp(angle, data_qubits[p_a], data_qubits[p_b])

    def swap_tracked(p_a, p_b):
        circuit.swap(data_qubits[p_a], data_qubits[p_b])
        p2l[p_a], p2l[p_b] = p2l[p_b], p2l[p_a]

    # --- 階段 1 ---
    for i in range(4):
        circuit.h(data_qubits[i])
        for j in range(i + 1, 4):
            cp_tracked(j, i)
    circuit.barrier()

    # --- 階段 2 ---
    for k in range(4):
        a = k
        b = 7 - k
        cp_tracked(a, b)
        swap_tracked(a, b)
        
        for m in range(4, b):
            cp_tracked(b, m)
        for n in range(a + 1, 4):
            cp_tracked(a, n)
            
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

    # --- 階段 3 ---
    for idx_i in range(4):
        phys_i = 3 - idx_i
        circuit.h(data_qubits[phys_i])
        for idx_j in range(idx_i + 1, 4):
            phys_j = 3 - idx_j
            cp_tracked(phys_j, phys_i)
    circuit.barrier()
    return p2l

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
    for scale in [1, 3, 5]:
        circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
        circ.compose(V_prep, qubits=data_qubits, inplace=True)
        circ.barrier()
        
        build_block_swap_qft_dd(circ, data_qubits, approx_degree=test_degree, zne_scale=scale, **builder_kwargs)
        
        circ.barrier()
        circ.append(inv_gate, data_qubits)
        circ.measure(data_qubits, cr_out)

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
    
    mitigated_fid = (15 * m3_fidelities[0] - 10 * m3_fidelities[1] + 3 * m3_fidelities[2]) / 8
    return min(max(mitigated_fid, 0), 1.0)

# ==========================================
# 2. 繪圖與核心處理架構
# ==========================================
def produce_2d_heatmap(target_degree, architecture, x_values, y_values, x_label, y_label, 
                       x_tick_format, y_tick_format, config_mapper, filename, title):
    print(f"[{title}] - {architecture} | Degree {target_degree}")
    fidelity_matrix = np.zeros((len(y_values), len(x_values)))
    
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            experiment_kwargs = config_mapper(x_val, y_val)
            
            if architecture == "Pure Telegate":
                fid = get_zne_mitigated_fidelity(test_degree=target_degree, shots=SHOTS, **experiment_kwargs)
            else:
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
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png', dpi=300)
    np.save(f'{OUTPUT_DIR}/{filename}.npy', fidelity_matrix)
    plt.close()

    return fidelity_matrix

def plot_combined_threshold_boundaries(matrices_by_degree, x_values, y_values, x_label, y_label, 
                                       x_tick_format, y_tick_format, filename, title, threshold=0.8):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap('tab10').colors
    legend_lines = []
    
    for deg, data in matrices_by_degree.items():
        color = colors[deg % len(colors)]
        rows, cols = data.shape
        legend_lines.append(mlines.Line2D([], [], color=color, lw=3, label=f'Degree {deg}'))
        
        for i in range(rows):
            for j in range(cols):
                if j < cols - 1 and (data[i, j] >= threshold) != (data[i, j+1] >= threshold):
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, lw=3)
                if i < rows - 1 and (data[i, j] >= threshold) != (data[i+1, j] >= threshold):
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, lw=3)

    _apply_plot_styling(ax, x_values, y_values, x_label, y_label, x_tick_format, y_tick_format, title, legend_lines)
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png', dpi=300)
    plt.close()

def plot_superimposed_threshold_boundaries(mat_pure_telegate, mat_bs, x_values, y_values, x_label, y_label, 
                                           x_tick_format, y_tick_format, filename, title, threshold=0.8):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap('tab10').colors
    legend_lines = []
    
    legend_lines.append(mlines.Line2D([], [], color='black', lw=2, linestyle='-', label='Pure Telegate Arch'))
    legend_lines.append(mlines.Line2D([], [], color='black', lw=2, linestyle='--', label='Block Swap Arch'))

    # 繪製 Pure Telegate (實線)
    for deg, data in mat_pure_telegate.items():
        color = colors[deg % len(colors)]
        legend_lines.append(mlines.Line2D([], [], color=color, lw=2, label=f'Degree {deg}'))
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                if j < cols - 1 and (data[i, j] >= threshold) != (data[i, j+1] >= threshold):
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, lw=2, linestyle='-')
                if i < rows - 1 and (data[i, j] >= threshold) != (data[i+1, j] >= threshold):
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, lw=2, linestyle='-')

    # 繪製 Block Swap (虛線)
    for deg, data in mat_bs.items():
        color = colors[deg % len(colors)]
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                if j < cols - 1 and (data[i, j] >= threshold) != (data[i, j+1] >= threshold):
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, lw=2, linestyle='--')
                if i < rows - 1 and (data[i, j] >= threshold) != (data[i+1, j] >= threshold):
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, lw=2, linestyle='--')

    _apply_plot_styling(ax, x_values, y_values, x_label, y_label, x_tick_format, y_tick_format, title, legend_lines)
    plt.savefig(f'{OUTPUT_DIR}/{filename}.png', dpi=300)
    plt.close()

def _apply_plot_styling(ax, x_values, y_values, x_label, y_label, x_tick_format, y_tick_format, title, legend_lines):
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([x_tick_format(x) for x in x_values])
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([y_tick_format(y) for y in y_values])
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(np.arange(-0.5, len(x_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_values), 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='--', linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    
    ax.legend(handles=legend_lines, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

# ==========================================
# 3. 實驗統整執行器 (單一綜合雜訊實驗)
# ==========================================
def run_combined_noise_experiment():
    # X: DD Ratio (0~1, step=0.1)
    x_vals = np.linspace(0.0, 1.0, STEPS)
    
    # Y: 綜合雜訊比例 (0.0 ~ 1.0)
    y_vals = np.linspace(0.0, 1.0, STEPS)
    
    MAX_1Q = 0.01
    MAX_2Q = 0.025
    MAX_SPAM = 0.025
    MAX_DEC_RATE = 0.05 # 對應 20us

    def get_combined_noise_params(y_scale):
        y_1q = y_scale * MAX_1Q
        y_2q = y_scale * MAX_2Q
        y_spam = y_scale * MAX_SPAM
        y_dec_rate = y_scale * MAX_DEC_RATE
        
        drift = (np.pi / 48) * (y_dec_rate / 0.02) if y_dec_rate > 0 else 0.0
        t_us = 1.0 / y_dec_rate if y_dec_rate > 0 else np.inf
        
        return y_1q, y_2q, y_spam, drift, t_us

    # Pure Telegate Mapper (已修正 M3 啟動邏輯與命名)
    def pure_telegate_map_combined(x_dd, y_scale):
        y_1q, y_2q, y_spam, drift, t_us = get_combined_noise_params(y_scale)
        return {
            'builder_func': build_test_circuit_dd,
            'builder_kwargs': {'dd_ratio': x_dd, 'phase_drift_per_epoch': drift},
            't1_us': t_us, 't2_us': t_us, 
            'err_1q': y_1q, 'err_2q': y_2q, 'spam_err': y_spam, 
            # 【關鍵修正】：當存在 SPAM 誤差時，給予對應的校正參數 (0.02) 以啟動 M3
            'mitigation_target_err': 0.02 if y_spam > 0 else 0.0 
        }

    # Block Swap Mapper
    def bs_map_combined(x_dd, y_scale):
        y_1q, y_2q, y_spam, drift, t_us = get_combined_noise_params(y_scale)
        return {
            'builder_kwargs': {'dd_ratio': x_dd, 'phase_drift_per_epoch': drift},
            't1_us': t_us, 't2_us': t_us, 
            'err_1q': y_1q, 'err_2q': y_2q, 'spam_err': y_spam
        }

    mat_pure_telegate = {}
    mat_bs = {}
    
    exp_name = "Combined All Noise Sources"
    exp_id = "Exp_AllNoise"
    
    x_label = "DD Ratio"
    y_label = "Combined Noise Scale (% of Max Noise)"
    x_fmt = lambda x: f"{x:.1f}"
    y_fmt = lambda y: f"{y*100:.0f}%"

    for deg in DEGREES_TO_TEST:
        mat_pure_telegate[deg] = produce_2d_heatmap(
            deg, "Pure Telegate", x_vals, y_vals, x_label, y_label, x_fmt, y_fmt,
            pure_telegate_map_combined, f"{exp_id}_PureTelegate_deg{deg}", f"Pure Telegate: {exp_name}"
        )
        
        mat_bs[deg] = produce_2d_heatmap(
            deg, "BlockSwap", x_vals, y_vals, x_label, y_label, x_fmt, y_fmt,
            bs_map_combined, f"{exp_id}_BlockSwap_deg{deg}", f"Block Swap: {exp_name}"
        )

    print(f"\n[{exp_name}] 繪製邊界圖中...")
    
    plot_combined_threshold_boundaries(
        mat_pure_telegate, x_vals, y_vals, x_label, y_label, x_fmt, y_fmt,
        f"{exp_id}_PureTelegate_Combined", f"Pure Telegate: {exp_name} (80% Threshold)"
    )
    plot_combined_threshold_boundaries(
        mat_bs, x_vals, y_vals, x_label, y_label, x_fmt, y_fmt,
        f"{exp_id}_BlockSwap_Combined", f"Block Swap: {exp_name} (80% Threshold)"
    )
    plot_superimposed_threshold_boundaries(
        mat_pure_telegate, mat_bs, x_vals, y_vals, x_label, y_label, x_fmt, y_fmt,
        f"{exp_id}_Superimposed_Combined", f"Comparison: {exp_name} (80% Threshold)"
    )

if __name__ == "__main__":
    print("====== 啟動雙架構 (Pure Telegate & Block Swap) 綜合雜訊評估實驗 ======\n")
    run_combined_noise_experiment()
    print("\n====== 所有實驗執行完畢！結果與比較圖已存入 results_combined_noise 目錄 ======")