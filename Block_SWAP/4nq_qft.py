import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, random_statevector, state_fidelity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def generate_pipeline_schedule(n_blocks):
    """
    動態脈動陣列排程器 (Systolic Array Scheduler)
    自動計算 N 塊晶片在管線化 QFT 中，每一個 Step 該做哪些平行運算。
    """
    # 每個資料塊的任務清單：先做 1 次 Local，再做 (n_blocks - 1 - i) 次向右的 Bridge
    tasks = [['LOCAL'] + ['BRIDGE'] * (n_blocks - 1 - i) for i in range(n_blocks)]
    
    data_pos = list(range(n_blocks))  # 紀錄每個資料塊現在在哪個實體晶片
    block_data = list(range(n_blocks)) # 紀錄每個實體晶片現在裝著哪個資料塊
    schedule = []
    
    while any(len(t) > 0 for t in tasks):
        busy_blocks = set()
        ops_this_step = []
        
        for i in range(n_blocks):
            if not tasks[i]:
                continue
                
            next_task = tasks[i][0]
            pos = data_pos[i]
            
            # 如果任務是 Local QFT，條件：必須身處在 B0 晶片，且 B0 當前空閒
            if next_task == 'LOCAL':
                if pos == 0 and pos not in busy_blocks:
                    busy_blocks.add(pos)
                    ops_this_step.append(('LOCAL', pos))
                    tasks[i].pop(0)
                    
            # 如果任務是向右 Bridge，條件：當前晶片與右邊相鄰晶片都空閒
            elif next_task == 'BRIDGE':
                if pos not in busy_blocks and (pos + 1) not in busy_blocks:
                    busy_blocks.add(pos)
                    busy_blocks.add(pos + 1)
                    ops_this_step.append(('BRIDGE', pos, pos + 1))
                    tasks[i].pop(0)
                    
                    # 模擬資料塊的實體位置交換
                    other_data = block_data[pos + 1]
                    block_data[pos], block_data[pos + 1] = block_data[pos + 1], block_data[pos]
                    data_pos[i] = pos + 1
                    data_pos[other_data] = pos
                    
        if ops_this_step:
            schedule.append(ops_this_step)
        else:
            break
            
    return schedule

def pipelined_dqft_N_qubit(qc, n_blocks):
    """
    N 塊晶片通用版分散式 QFT (管線化 + 自動排程)
    每塊晶片包含 4 顆 Qubit
    """
    n_qubits = n_blocks * 4
    p2l = list(range(n_qubits))

    def cp_tracked(p_a, p_b):
        log_a = p2l[p_a]
        log_b = p2l[p_b]
        angle = np.pi / (2 ** abs(log_a - log_b))
        qc.cp(angle, p_a, p_b)

    def swap_tracked(p_a, p_b):
        qc.swap(p_a, p_b)
        p2l[p_a], p2l[p_b] = p2l[p_b], p2l[p_a]

    def local_qft_tracked(block):
        for i in range(4):
            qc.h(block[i])
            for j in range(i + 1, 4):
                cp_tracked(block[j], block[i])

    def pipelined_bridge(block_A, block_B):
        for k in range(4):
            a = block_A[k]
            b = block_B[k]
            cp_tracked(a, b)
            swap_tracked(a, b)
            
            for m in range(k + 1, 4):
                cp_tracked(b, block_B[m])
            for n in range(k + 1, 4):
                cp_tracked(a, block_A[n])

    def local_reverse_tracked(block):
        swap_tracked(block[0], block[3])
        swap_tracked(block[1], block[2])

    qc.barrier(label=f"Start N-Block D-QFT ({n_blocks} Blocks, {n_qubits} Qubits)")

    # 1. 取得自動排程計畫
    schedule = generate_pipeline_schedule(n_blocks)

    # 2. 根據計畫，按 Step 執行平行運算
    for step_idx, step_ops in enumerate(schedule):
        # 標記目前正在進行的平行輪次
        op_labels = []
        for op in step_ops:
            if op[0] == 'LOCAL': op_labels.append(f"L(B{op[1]})")
            else: op_labels.append(f"Bridge(B{op[1]}-B{op[2]})")
        qc.barrier(label=f"Step {step_idx+1}: " + " | ".join(op_labels))
        
        # 執行該輪次的所有操作
        for op in step_ops:
            if op[0] == 'LOCAL':
                b_idx = op[1]
                block_qubits = list(range(b_idx * 4, b_idx * 4 + 4))
                local_qft_tracked(block_qubits)
            elif op[0] == 'BRIDGE':
                b1_idx, b2_idx = op[1], op[2]
                block_A = list(range(b1_idx * 4, b1_idx * 4 + 4))
                block_B = list(range(b2_idx * 4, b2_idx * 4 + 4))
                pipelined_bridge(block_A, block_B)

    # 3. 階段：全 QPU 局部翻轉 (Local Reversal)
    qc.barrier(label="Local Reversal on EVERY QPU")
    for b_idx in range(n_blocks):
        block_qubits = list(range(b_idx * 4, b_idx * 4 + 4))
        local_reverse_tracked(block_qubits)

    return p2l

# ==========================================
# 嚴格驗證主程式 (以 4 塊晶片 / 16 Qubits 為例 - 記憶體優化版)
# ==========================================
n_blocks = 4 
n_qubits = n_blocks * 4

print("="*65)
print(f" 正在進行 {n_qubits}-Qubit ({n_blocks} 塊晶片) 通用自動排程版數值驗證...")
print("="*65)

# 產生初始狀態向量 (這個向量本身很小，只有 1MB 左右)
state_initial = random_statevector(2**n_qubits)

# 1. 建立「純運算」的量子電路
qc_verify = QuantumCircuit(n_qubits)

# 2. 執行自動擴充版管線化路由
pipelined_dqft_N_qubit(qc_verify, n_blocks)

# 3. 加上無敵的驗證反函數
inverse_qft = QFT(num_qubits=n_qubits, do_swaps=True).reverse_bits().inverse()
qc_verify.compose(inverse_qft, inplace=True)

# 4. 直接讓初始向量通過電路進行「演化」
# 這樣 Qiskit 就不會去建立 64GB 的重置矩陣，而是有效率地進行稀疏矩陣向量乘法！
state_final = state_initial.evolve(qc_verify)

# 計算保真度
fidelity = state_fidelity(state_initial, state_final)

print(f" {n_qubits}-Qubit Fidelity: {fidelity:.15f}")


# ==========================================
# 輸出高解析度電路圖
# ==========================================
'''
qc_draw = QuantumCircuit(n_qubits, name=f"Pipelined_Ultimate_D-QFT_{n_qubits}")
pipelined_dqft_N_qubit(qc_draw, n_blocks)

try:
    fig = qc_draw.draw(output='mpl', style='clifford', fold=-1, scale=0.3)
    for text in fig.axes[0].texts:
        if 'label' in text.__dict__['_text']:
            text.set_fontsize(7)
    
    filename = f"{n_qubits}_qubit_pipelined_universal_dqft.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 高解析度電路圖已成功儲存為：{filename}")
    plt.show()
except Exception as e:
    print(f"\n圖形輸出失敗。錯誤訊息：{e}")
    '''