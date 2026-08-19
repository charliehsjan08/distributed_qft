import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, random_statevector, state_fidelity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def pipelined_dqft_12q_tracked_ultimate(qc):
    """
    12-qubit 分散式 QFT (管線化排程)
    """
    p2l = list(range(12))

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

    B0 = [0, 1, 2, 3]
    B1 = [4, 5, 6, 7]
    B2 = [8, 9, 10, 11]

    qc.barrier(label="Start Pipelined D-QFT")

    # 階段 1：Local QFT on B0
    qc.barrier(label="Local QFT on Block 0 (Batch 1)")
    local_qft_tracked(B0)

    # 階段 2：B0 與 B1 管線化交會
    qc.barrier(label="Pipeline Round 1: Cross B0 and B1")
    pipelined_bridge(B0, B1)

    # ==========================================
    # 🔥 你的極致排程優化：階段 3 與 4 對調！
    # 讓邏輯 0~3 繼續往 B2 推進，同時 B0 可以平行處理邏輯 4~7！
    # ==========================================
    
    # 階段 3 (原階段 4)：B1 與 B2 管線化交會
    qc.barrier(label="Pipeline Round 2: Cross B1 and B2")
    pipelined_bridge(B1, B2)

    # 階段 4 (原階段 3)：Local QFT on B0 (承載第二批資料)
    qc.barrier(label="Local QFT on Block 0 (Batch 2)")
    local_qft_tracked(B0)
    
    # ==========================================

    # 階段 5：B0 與 B1 最後交會
    qc.barrier(label="Pipeline Round 3: Cross B0 and B1")
    pipelined_bridge(B0, B1)

    # 階段 6：收尾 Local QFT
    qc.barrier(label="Final Local QFT on Block 0 (Batch 3)")
    local_qft_tracked(B0)

    # 階段 7：全 QPU 局部翻轉 (Local Reversal)
    qc.barrier(label="Local Reversal on EVERY QPU")
    local_reverse_tracked(B0)
    local_reverse_tracked(B1)
    local_reverse_tracked(B2)

    return p2l


# ==========================================
# 嚴格驗證主程式
# ==========================================
print("="*55)
print(" 正在進行 12-Qubit D-QFT 數值驗證...")
print("="*55)

n_qubits = 12
state_initial = random_statevector(2**n_qubits)

qc_verify = QuantumCircuit(n_qubits)
qc_verify.initialize(state_initial, qc_verify.qubits)

# 執行管線化路由
pipelined_dqft_12q_tracked_ultimate(qc_verify)

# 🔥 你的優雅驗證法：完美利用空間對稱性與無 SWAP 反轉！
# 完全不用傳入 qubits mapping，直接在原生空間中對消
inverse_qft = QFT(num_qubits=n_qubits, do_swaps=True).reverse_bits().inverse()
qc_verify.compose(inverse_qft, inplace=True)

state_final = Statevector(qc_verify)
fidelity = state_fidelity(state_initial, state_final)

print(f"12-qubit D-QFT Fidelity: {fidelity:.15f}")


# ==========================================
# 輸出高解析度電路圖
# ==========================================
'''
qc_draw = QuantumCircuit(n_qubits, name="Pipelined_Ultimate_D-QFT_12")
pipelined_dqft_12q_tracked_ultimate(qc_draw)

try:
    fig = qc_draw.draw(output='mpl', style='clifford', fold=-1, scale=0.45)
    for text in fig.axes[0].texts:
        if 'label' in text.__dict__['_text']:
            text.set_fontsize(8)
    
    filename = "12_qubit_pipelined_ultimate_dqft.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 高解析度電路圖已成功儲存為：{filename}")
    plt.show()
except Exception as e:
    print(f"\n圖形輸出失敗。錯誤訊息：{e}")
    '''