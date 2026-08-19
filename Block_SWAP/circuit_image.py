import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# ==========================================
# 準備 8-qubit (2塊 QPU, 每塊4顆) 的電路生成器
# ==========================================

# 1. 傳統方法 (Pure Telegate) 8-qubit QFT 電路
def create_pure_telegate_qft_8():
    qc = QuantumCircuit(8, name="Pure_Telegate_QFT")
    for i in range(8):
        qc.h(i)
        for j in range(i+1, 8):
            angle = np.pi / (2**(j-i))
            qc.cp(angle, j, i)  # 包含大量跨 QPU 邊界的遠端 CP
    return qc

# 2. 新方法：單純展示 Block SWAP 路由核心
def create_block_swap_routing_8():
    qc = QuantumCircuit(8, name="Block_SWAP_Routing")
    P0 = [0, 1, 2, 3]
    P1 = [4, 5, 6, 7]
    
    qc.barrier(label="Cross-Block CP & Block SWAP")
    # 模擬 4 顆 qubit 的整體區塊交換
    for i in range(4):
        qc.swap(P0[i], P1[i])
    return qc

# 3. 新方法：完整的 Pipelined Block-Routing QFT
def create_pipelined_routing_qft_8():
    qc = QuantumCircuit(8, name="Pipelined_Block_Routing_QFT")
    P0 = [0, 1, 2, 3]  # QPU A
    P1 = [4, 5, 6, 7]  # QPU B

    # 暖身：Local QFT on block 0
    qc.barrier(label="Local Qft (Block 0)")
    for i in range(4):
        qc.h(P0[i])
        for j in range(i+1, 4):
            qc.cp(np.pi / (2**(j-i)), P0[j], P0[i])

    # 跨界互動與 Block SWAP
    qc.barrier(label="Cross CP & Block SWAP")
    for i in range(4):
        for j in range(4):
            angle = np.pi / (2**((j+4) - i))
            qc.cp(angle, P1[j], P0[i])
        qc.swap(P0[i], P1[i])

    # 收尾：Local QFT on swapped block
    qc.barrier(label="Local Qft (Block 1)")
    for i in range(4):
        qc.h(P0[i])
        for j in range(i+1, 4):
            qc.cp(np.pi / (2**(j-i)), P0[j], P0[i])
            
    return qc

# ==========================================
# 繪製並儲存這三張圖表供 PPT 使用
# ==========================================

circuits = {
    "Figure_1_Pure_Telegate_QFT.png": create_pure_telegate_qft_8(),
    "Figure_2_Block_SWAP_Core.png": create_block_swap_routing_8(),
    "Figure_3_Pipelined_Routing_QFT.png": create_pipelined_routing_qft_8()
}

for filename, circ in circuits.items():
    # 使用 mpl 模式畫出乾淨、適合簡報的圖表
    fig = circ.draw(output='mpl', style='clifford', fold=-1)
    fig.suptitle(f"8-Qubit (2-QPU) Distributed QFT: {circ.name}", fontsize=14, y=1.03)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 成功生成並儲存：{filename}")