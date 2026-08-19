import os
import numpy as np
import matplotlib.pyplot
matplotlib.use('Agg') # Force non-interactive backend before importing pyplot
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

# 引入我們封裝好的核心 Harness
from qft_experiment_harness import (
    get_zne_mitigated_fidelity, 
    build_test_circuit_dd
)

# ==========================================
# 執行環境設定 (Cluster vs Local)
# ==========================================
# 在本地測試時設為 True，只會跑 Degree=4，並使用極小的網格與 Shots 來快速驗證程式是否會報錯。
# 上傳到 nano5 前，請將此改為 False。
LOCAL_TEST_MODE = False

if LOCAL_TEST_MODE:
    print("⚠️ 運行在 LOCAL TEST MODE (快速驗證)")
    DEGREES_TO_TEST = [4]
    STEPS = 3
    SHOTS = 100
else:
    print("🚀 運行在 FULL CLUSTER MODE")
    DEGREES_TO_TEST = list(range(8)) # 0 到 7
    STEPS = 11                       # 依照你的要求，10 steps 加上 0，共 11 點
    SHOTS = 3000

# 建立輸出目錄
os.makedirs("results", exist_ok=True)


# ==========================================
# 共用繪圖與執行核心
# ==========================================
def produce_2d_heatmap(target_degree, x_values, y_values, x_label, y_label, 
                       x_tick_format, y_tick_format, config_mapper, filename, title):
    print(f"[{title}] - Degree {target_degree} (Grid: {len(x_values)}x{len(y_values)})")
    fidelity_matrix = np.zeros((len(y_values), len(x_values)))
    
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            # 1. 取得該座標的參數設定
            experiment_kwargs = config_mapper(x_val, y_val)
            # 2. 執行 ZNE 流程
            fid = get_zne_mitigated_fidelity(test_degree=target_degree, shots=SHOTS, **experiment_kwargs)
            # 3. 儲存結果
            fidelity_matrix[i, j] = fid
            print(f"  └─ X: {x_tick_format(x_val):>5} | Y: {y_tick_format(y_val):>7} -> Fidelity: {fid*100:5.1f}%")

    # 繪製圖表
    plt.figure(figsize=(10, 8))
    dx = (x_values[1] - x_values[0]) / 2 if len(x_values) > 1 else 0.5
    dy = (y_values[1] - y_values[0]) / 2 if len(y_values) > 1 else 0.5
    extent = [x_values[0]-dx, x_values[-1]+dx, y_values[0]-dy, y_values[-1]+dy]
    
    im = plt.imshow(fidelity_matrix * 100, cmap='plasma', aspect='auto', 
                    origin='lower', extent=extent, vmin=0, vmax=100)
    
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
    plt.savefig(f'results/{filename}.png', dpi=300)

    np.save(f'results/{filename}.npy', fidelity_matrix)
    
    plt.close()

    return fidelity_matrix

# ==========================================
# 實驗 1: 1Q Gate Error vs DD Ratio
# ==========================================
def run_exp1_1q_vs_dd():
    # X: DD Ratio (0~1, step=0.1)
    # Y: 1Q Error (0~1%, step=0.1%)
    x_vals = np.linspace(0.0, 1.0, STEPS)
    y_vals = np.linspace(0.0, 0.01, STEPS)

    def mapper(x_dd, y_1q_err):
        return {
            'builder_func': build_test_circuit_dd,
            'builder_kwargs': {'dd_ratio': x_dd, 'phase_drift_per_epoch': 0.0}, # Drift=0, 只有Pulse Error
            't1_us': np.inf, 't2_us': np.inf,
            'err_1q': y_1q_err, 'err_2q': 0.0, 'spam_err': 0.0, 'mitigation_target_err': 0.0
        }
    mat_1q = {}
    for deg in DEGREES_TO_TEST:
        mat_1q[deg]=produce_2d_heatmap(
            deg, x_vals, y_vals, "DD Ratio", "1Q Gate Error (%)",
            lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%",
            mapper, f"Exp1_1Q_vs_DD_deg{deg}", "1Q Error vs DD Ratio"
        )

    plot_combined_threshold_boundaries(
        mat_1q, x_vals, y_vals, "DD Ratio", "1Q Gate Error (%)",
        lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%",
        "Exp1_1Q_vs_DD_Combined", "1Q Error vs DD Ratio (80% Threshold)", "results"
    )


# ==========================================
# 實驗 2: 2Q Gate Error vs DD Ratio
# ==========================================
def run_exp2_2q_vs_dd():
    # X: DD Ratio (0~1, step=0.1)
    # Y: 2Q Error (0~2.5%, step=0.25%)
    x_vals = np.linspace(0.0, 1.0, STEPS)
    y_vals = np.linspace(0.0, 0.025, STEPS)

    def mapper(x_dd, y_2q_err):
        return {
            'builder_func': build_test_circuit_dd,
            'builder_kwargs': {'dd_ratio': x_dd, 'phase_drift_per_epoch': 0.0}, # Drift=0, 證明DD在純2Q錯下無作用
            't1_us': np.inf, 't2_us': np.inf,
            'err_1q': 0.0, 'err_2q': y_2q_err, 'spam_err': 0.0, 'mitigation_target_err': 0.0
        }
    mat_2q = {}
    for deg in DEGREES_TO_TEST:
        mat_2q[deg] =produce_2d_heatmap(
            deg, x_vals, y_vals, "DD Ratio", "2Q Gate Error (%)",
            lambda x: f"{x:.1f}", lambda y: f"{y*100:.2f}%",
            mapper, f"Exp2_2Q_vs_DD_deg{deg}", "2Q Error vs DD Ratio"
        )
    plot_combined_threshold_boundaries(
        mat_2q, x_vals, y_vals, "DD Ratio", "2Q Gate Error (%)",
        lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%",
        "Exp2_2Q_vs_DD_Combined", "2Q Error vs DD Ratio (80% Threshold)", "results"
    )

# ==========================================
# 實驗 3: SPAM Error vs DD Ratio
# ==========================================
def run_exp3_spam_vs_dd():
    # X: DD Ratio (0~1, step=0.1)
    # Y: SPAM Error (0~2.5%, step=0.25%)
    x_vals = np.linspace(0.0, 1.0, STEPS)
    y_vals = np.linspace(0.0, 0.025, STEPS)

    def mapper(x_dd, y_spam_err):
        return {
            'builder_func': build_test_circuit_dd,
            'builder_kwargs': {'dd_ratio': x_dd, 'phase_drift_per_epoch': 0.0},
            't1_us': np.inf, 't2_us': np.inf,
            'err_1q': 0.0, 'err_2q': 0.0, 'spam_err': y_spam_err, 
            'mitigation_target_err': 0.0 # 強制設為0，讓M3不發揮作用，以顯示Raw SPAM破壞力
        }
    mat_spam = {}
    for deg in DEGREES_TO_TEST:
        mat_spam[deg] = produce_2d_heatmap(
            deg, x_vals, y_vals, "DD Ratio", "SPAM Error (%)",
            lambda x: f"{x:.1f}", lambda y: f"{y*100:.2f}%",
            mapper, f"Exp3_SPAM_vs_DD_deg{deg}", "Raw SPAM vs DD Ratio"
        )

    plot_combined_threshold_boundaries(
        mat_spam, x_vals, y_vals, "DD Ratio", "SPAM Error (%)",
        lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%",
        "Exp3_SPAM_vs_DD_Combined", "Raw SPAM vs DD Ratio (80% Threshold)", "results"
    )


# ==========================================
# 實驗 4: Decoherence (T1/T2) vs DD Ratio
# ==========================================
def run_exp4_decoherence_vs_dd():
    # X: DD Ratio (0~1, step=0.1)
    # Y: Decoherence Rate (0 ~ 0.05 1/us, 對應 inf ~ 20us)
    x_vals = np.linspace(0.0, 1.0, STEPS)
    y_vals = np.linspace(0.0, 0.05, STEPS)

    def mapper(x_dd, y_rate):
        current_T = 1.0 / y_rate if y_rate > 0 else np.inf
        # Coherence time 下降時，Phase drift 也成正比上升 (給DD發揮的空間)
        drift = (np.pi / 48) * (y_rate / 0.02) if y_rate > 0 else 0.0

        return {
            'builder_func': build_test_circuit_dd,
            'builder_kwargs': {'dd_ratio': x_dd, 'phase_drift_per_epoch': drift},
            't1_us': current_T, 't2_us': current_T,
            'err_1q': 0.0, 'err_2q': 0.0, 'spam_err': 0.0, 'mitigation_target_err': 0.0
        }
    mat_dec = {}
    for deg in DEGREES_TO_TEST:
        mat_dec[deg] = produce_2d_heatmap(
            deg, x_vals, y_vals, "DD Ratio", r"Decoherence Rate $\Gamma = 1/T$ ($1/\mu s$)",
            lambda x: f"{x:.1f}", lambda y: f"inf" if y==0 else f"{int(1.0/y)}us",
            mapper, f"Exp4_Decoherence_vs_DD_deg{deg}", "Decoherence (T1/T2) vs DD Ratio"
        )

    plot_combined_threshold_boundaries(
        mat_dec, x_vals, y_vals, "DD Ratio", r"Decoherence Rate $\Gamma = 1/T$ ($1/\mu s$)",
        lambda x: f"{x:.1f}", lambda y: f"{y*100:.1f}%",
        "Exp4_Decoherence_vs_DD_Combined", "Decoherence Rate vs DD Ratio (80% Threshold)", "results"
    )

if __name__ == "__main__":
    print("====== 啟動實驗批次執行序列 ======\n")
    
    # 你可以把不想跑的先註解掉
    run_exp1_1q_vs_dd()
    run_exp2_2q_vs_dd()
    run_exp3_spam_vs_dd()
    run_exp4_decoherence_vs_dd()
    
    print("\n====== 所有實驗執行完畢！ ======")