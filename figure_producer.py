import numpy as np
import matplotlib.pyplot as plt

# Import the core tools from your newly packaged harness
from qft_experiment_harness import (
    get_zne_mitigated_fidelity, 
    build_test_circuit_dd, 
    build_test_circuit_hybrid
)

def produce_2d_heatmap(
    target_degree,
    x_values, 
    y_values, 
    x_label, 
    y_label,
    x_tick_format,
    y_tick_format,
    config_mapper
):
    """
    Generates a 2D fidelity phase diagram based on custom sweeping parameters.
    
    Args:
        target_degree (int): The AQFT approximation degree to test.
        x_values (array): Values to sweep across the X-axis.
        y_values (array): Values to sweep across the Y-axis.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        x_tick_format (lambda): Function to format X tick strings (e.g., lambda x: f"{x*100}%").
        y_tick_format (lambda): Function to format Y tick strings.
        config_mapper (callable): A function that takes (x, y) and returns a dict of 
                                  kwargs to pass into get_zne_mitigated_fidelity.
    """
    print(f"\n=== Generating 2D Heatmap for Degree {target_degree} ===")
    
    fidelity_matrix = np.zeros((len(y_values), len(x_values)))
    
    for i, y_val in enumerate(y_values):
        print(f"Sweeping Y-axis level: {y_tick_format(y_val)}")
        for j, x_val in enumerate(x_values):
            
            # 1. Map the current (X, Y) coordinates to the hardware/circuit parameters
            experiment_kwargs = config_mapper(x_val, y_val)
            
            # 2. Execute the ZNE pipeline
            fid = get_zne_mitigated_fidelity(test_degree=target_degree, **experiment_kwargs)
            
            # 3. Store result
            fidelity_matrix[i, j] = fid
            print(f"  -> X: {x_tick_format(x_val)} | Fidelity: {fid*100:.1f}%")

    # ==========================================
    # PLOTTING LOGIC
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # Calculate extents to center the blocks perfectly over the ticks
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
    plt.title(f'Phase Diagram (Degree {target_degree})', fontsize=14)
    
    # Overlay numerical text
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            val = fidelity_matrix[i, j] * 100
            text_color = "black" if val > 70 else "white"
            plt.text(x_values[j], y_values[i], f'{val:.1f}', 
                     ha="center", va="center", color=text_color, fontsize=10)
                     
    plt.savefig(f'custom_heatmap_output.png', dpi=300)        #file name
    print("Done! Saved to 'custom_heatmap_output.png'.\n")


# ==========================================
# IMPLEMENTATION / TUNING SECTION
# ==========================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # STEP 1: Define your axes and what you want to sweep
    # ---------------------------------------------------------
    X_ARRAY = np.linspace(0.0, 1.0, 5)            # Example: DD Ratio from 0% to 100%
    Y_ARRAY = np.arange(0.0, 0.026, 0.005)        # Example: Error Rate from 0.0% to 2.5%
    
    # ---------------------------------------------------------
    # STEP 2: Define how the (X, Y) maps to the experiment logic
    # ---------------------------------------------------------
    def my_custom_mapper(x_val, y_val):
        """
        In this example: 
        X-axis controls the DD Ratio.
        Y-axis controls the 2Q Gate Error.
        Everything else is set to ideal/perfect.
        """
        # Calculate phase drift if you want to tie it to the error, or set to 0.0
        drift = (np.pi / 48) * (y_val / 0.01) if y_val > 0 else 0.0
        
        return {
            'shots': 3000,
            
            # --- Architecture Configuration ---
            'builder_func': build_test_circuit_dd,
            'builder_kwargs': {
                'dd_ratio': x_val,                # Linked to X-axis
                'phase_drift_per_epoch': drift
            },
            
            # --- Hardware Noise Configuration ---
            't1_us': np.inf,                      # Perfect
            't2_us': np.inf,                      # Perfect
            'err_1q': 0.0,                        # Perfect
            'err_2q': y_val,                      # Linked to Y-axis
            'spam_err': 0.0,                      # Perfect
            'mitigation_target_err': 0.0
        }

    # ---------------------------------------------------------
    # STEP 3: Run the producer
    # ---------------------------------------------------------
    produce_2d_heatmap(
        target_degree=4,
        x_values=X_ARRAY,
        y_values=Y_ARRAY,
        x_label="Dynamical Decoupling Ratio",
        y_label="2-Qubit Gate Error",
        x_tick_format=lambda x: f"{int(x*100)}%",
        y_tick_format=lambda y: f"{y*100:.1f}%",
        config_mapper=my_custom_mapper
    )