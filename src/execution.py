from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2
from qiskit.transpiler import Target
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.controlflow import IfElseOp

def run_local_simulation(circuit, noise_model=None, shots=10000):
    """Executes a target circuit configuration locally on the AerSimulator backend."""
    if noise_model:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()
        
    # 1. Add 'if_else' to the basis gates and map it in the custom dictionary
    custom_target = Target.from_configuration(
        basis_gates=['cx', 'rz', 'sx', 'x', 'h', 'z', 'measure', 'reset', 'unitary', 'if_else'],
        custom_name_mapping={
            'unitary': UnitaryGate,
            'if_else': IfElseOp
        }
    )
    
    # 2. Transpile using the fully capable custom target
    compiled_circuit = transpile(circuit, target=custom_target, optimization_level=0)
    
    sampler = BackendSamplerV2(backend=simulator)
    sampler.options.default_shots = shots
    
    job = sampler.run([compiled_circuit])
    return job.result()[0].data.c_out.get_counts()