def correct_crossing_cx(n_blocks):
    """
    修正版：只計算真正跨越 QPU 邊界（佔用通訊通道）的 Crossing CX 數量
    """
    # 1. 老方法：所有跨界 CP 都要走 Telegate，每個 CP 拆成 2 個 Crossing CX
    total_qubits = n_blocks * 4
    cross_cp_total_pairs = 0
    for i in range(total_qubits):
        for j in range(i + 1, total_qubits):
            if (i // 4) != (j // 4):
                cross_cp_total_pairs += 1
    old_crossing_cx = cross_cp_total_pairs * 2

    # 2. 新方法：只有 Block SWAP 的過程需要跨越邊界！
    # 總共需要進行的區塊相遇回合數 = n*(n-1)/2
    total_encounters = n_blocks * (n_blocks - 1) // 2
    
    # 每次相遇我們執行一次 Block SWAP (4顆 qubit 各做一次 SWAP)
    # 每個跨晶片 SWAP 消耗 3 個 Crossing CX
    new_crossing_cx = total_encounters * 4 * 5

    return old_crossing_cx, new_crossing_cx

print("=" * 65)
print(f"{'QPU 數量 (n)':<12} | {'總 Qubits':<10} | {'老方法 Crossing CX':<22} | {'新方法 Crossing CX':<22}")
print("=" * 65)

for n in range(2,12):
    old_cx, new_cx = correct_crossing_cx(n)
    print(f"n = {n:<9} | {n*4:<10} | {old_cx:<22} | {new_cx:<22}")

print("=" * 65)