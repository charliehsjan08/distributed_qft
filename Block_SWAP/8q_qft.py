import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, random_statevector, state_fidelity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def distributed_qft_8qubit_symmetric_tracked(qc):
    """
    自訂分散式 8-qubit QFT (對稱路由 + 動態追蹤版)
    """
    # 建立「實體到邏輯」追蹤表
    p2l = list(range(8))
    
    def cp_tracked(p_a, p_b):
        log_a = p2l[p_a]
        log_b = p2l[p_b]
        angle = np.pi / (2 ** abs(log_a - log_b))
        qc.cp(angle, p_a, p_b)

    def swap_tracked(p_a, p_b):
        qc.swap(p_a, p_b)
        p2l[p_a], p2l[p_b] = p2l[p_b], p2l[p_a]

    qc.barrier(label="Start Distributed QFT (8Q)")
    
    # ==========================================
    # Phase 1: Local A
    # ==========================================
    for i in range(4):
        qc.h(i)
        for j in range(i+1, 4):
            cp_tracked(j, i)
            
    qc.barrier(label="Symmetric Bridge Routing")

    # ==========================================
    # Phase 2: 對稱橋接 (你的天才迴圈，完美產生 16 對 CP)
    # ==========================================
    for k in range(4):
        a = k
        b = 7 - k
        
        # 1. 橋上 Telegate CP
        cp_tracked(a, b)
        
        # 2. 過獨木橋 (Network SWAP)
        swap_tracked(a, b)
        
        # 3. 邏輯 a 與剩下的做 Local CP
        for m in range(4, b):
            cp_tracked(b, m)
            
        # 4. 邏輯 b 與剩下的做 Local CP
        for n in range(a+1, 4):
            cp_tracked(a, n)

    qc.barrier(label="Local B QFT (Reversed on Phys A)")

    # ==========================================
    # Phase 3: Local B (反向降落版)
    # 利用 Tracker，這裡完全不需要手動算邏輯對應！
    # ==========================================
    for idx_i in range(4):
        phys_i = 3 - idx_i
        qc.h(phys_i)
        for idx_j in range(idx_i + 1, 4):
            phys_j = 3 - idx_j
            cp_tracked(phys_j, phys_i)
            
    return p2l


# ==========================================
# 嚴格驗證主程式 (Statevector Fidelity)
# ==========================================
print("="*55)
print(" 正在進行 8-Qubit D-QFT 數值驗證...")
print("="*55)

n_qubits = 8
state_initial = random_statevector(2**n_qubits)

# 驗證用電路
qc_verify = QuantumCircuit(n_qubits)
qc_verify.initialize(state_initial, qc_verify.qubits)

# 執行對稱版路由
final_mapping = distributed_qft_8qubit_symmetric_tracked(qc_verify)

# 【關鍵修正】：避開 Qiskit 的 do_swaps 陷阱！
# 由於演算法結束時，資料是完美的全局顛倒 (0~7 變為 7~0)
# 我們在驗證層手動把它翻正，這樣就能直接套用最標準的反函數來做純數學驗證
#qc_verify.barrier(label="Validation Block (Restore Order)")
#for i in range(4):
    #qc_verify.swap(i, 7-i)

# 使用最標準的、包含 swaps 的 QFT 反函數
inverse_qft = QFT(num_qubits=n_qubits, do_swaps=True).reverse_bits().inverse()
qc_verify.compose(inverse_qft, inplace=True)

# 測量保真度
state_final = Statevector(qc_verify)
fidelity = state_fidelity(state_initial, state_final)

print(f"8-qubit D-QFT Fidelity: {fidelity:.15f}")


# ==========================================
# 繪圖主程式：單純打印路由核心
# ==========================================
'''
print("="*55)
print(" 正在生成 8-Qubit 對稱優化分散式 QFT 電路圖...")
print("="*55)

qc_draw = QuantumCircuit(n_qubits, name="Tracked_Symmetric_D-QFT_8")
distributed_qft_8qubit_symmetric_tracked(qc_draw)

try:
    fig = qc_draw.draw(output='mpl', style='clifford', fold=-1, scale=0.7)
    for text in fig.axes[0].texts:
        if 'label' in text.__dict__['_text']:
            text.set_fontsize(10)
    
    filename = "8_qubit_symmetric_tracked_dqft.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 高解析度電路圖已成功儲存為：{filename}")
    plt.show()
    
except Exception as e:
    print(f"\n圖形輸出失敗。錯誤訊息：{e}")
    '''