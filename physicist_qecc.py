import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from qiskit.quantum_info import StabilizerState, Statevector, PauliList

def get_5q_encoder_circuit():
    qr_d = QuantumRegister(5, 'q')
    encode_0 = QuantumCircuit(qr_d, name='user_encoder_v18')
    
    # --- START OF USER CODE (FIXED) ---
    
    # Layer 1: Hadamards
    encode_0.h([qr_d[0], qr_d[1], qr_d[2], qr_d[3]])
    
    # Layer 2: CNOTs to q4 (using .cx)
    encode_0.cx(qr_d[0], qr_d[4])
    encode_0.cx(qr_d[1], qr_d[4])
    encode_0.cx(qr_d[2], qr_d[4])
    encode_0.cx(qr_d[3], qr_d[4])
    encode_0.barrier()
    
    # Layer 3: CNOT cycle (using .cx)
    encode_0.cz(qr_d[0], qr_d[1])
    encode_0.cz(qr_d[1], qr_d[2])
    encode_0.cz(qr_d[2], qr_d[3])
    encode_0.cz(qr_d[3], qr_d[4])
    
    encode_0.barrier()
    
    encode_0.cz(qr_d[4], qr_d[0])
    return encode_0


def get_baseline_code():
    n_data = 1
    qr_d = QuantumRegister(n_data, 'q')
    cr_l = ClassicalRegister(1, 'logical_out')
    
    # 1. Encode |0> and add an ID gate for noise
    encode_0 = QuantumCircuit(qr_d, name='encode_base')
    # for i in range(40):
    #   encode_0.id(qr_d[0]) 
    #   encode_0.h(qr_d[0])
    #   encode_0.h(qr_d[0])
    
    # 2. Syndrome measurement (none)
    syndrome_measure = QuantumCircuit(qr_d, name='syndrome_base')
    
    # 3. Correction table (none)
    correction_table = {}
    
    # 4. Logical Z measure (just measure Z)
    logical_z_measure = QuantumCircuit(qr_d, cr_l, name='measure_base')
    logical_z_measure.measure(qr_d[0], cr_l[0])
    
    return {
        'name': 'Baseline (No QEC)',
        'n_data': n_data,
        'n_ancilla': 0,
        'n_logical_z_anc': 0,
        'encode_0': encode_0,
        'syndrome_measure': syndrome_measure,
        'correction_table': correction_table,
        'logical_z_measure': logical_z_measure
    }

def get_5q_code():
    n_data = 5
    n_ancilla = 4
    n_logical_z_anc = 1
    
    qr_d = QuantumRegister(n_data, 'q')
    qr_a = QuantumRegister(n_ancilla, 'a')
    cr_s = ClassicalRegister(n_ancilla, 'syn')
    qr_lz = QuantumRegister(n_logical_z_anc, 'lz')
    cr_l = ClassicalRegister(1, 'logical_out')
    
    # 1. Encode |0>_L (from user's fixed function)
    encode_0 = get_5q_encoder_circuit()

    # 2. Syndrome Measurement Circuit
    # This circuit IS for the 'XZZXI' stabilizers,
    # which MATCHES the user's encoder.
    syndrome_measure = QuantumCircuit(qr_d, qr_a, cr_s, name='syndrome_5q_XZZXI')
    syndrome_measure.h(qr_a)
    # S0 = X Z Z X I
    syndrome_measure.cx(qr_a[0], qr_d[0])
    syndrome_measure.cz(qr_a[0], qr_d[1])
    syndrome_measure.cz(qr_a[0], qr_d[2])
    syndrome_measure.cx(qr_a[0], qr_d[3])
    # S1 = I X Z Z X
    syndrome_measure.cx(qr_a[1], qr_d[1])
    syndrome_measure.cz(qr_a[1], qr_d[2])
    syndrome_measure.cz(qr_a[1], qr_d[3])
    syndrome_measure.cx(qr_a[1], qr_d[4])
    # S2 = X I X Z Z
    syndrome_measure.cx(qr_a[2], qr_d[0])
    syndrome_measure.cx(qr_a[2], qr_d[2])
    syndrome_measure.cz(qr_a[2], qr_d[3])
    syndrome_measure.cz(qr_a[2], qr_d[4])
    # S3 = Z X I X Z
    syndrome_measure.cz(qr_a[3], qr_d[0])
    syndrome_measure.cx(qr_a[3], qr_d[1])
    syndrome_measure.cx(qr_a[3], qr_d[3])
    syndrome_measure.cz(qr_a[3], qr_d[4])
    syndrome_measure.h(qr_a)
    syndrome_measure.measure(qr_a, cr_s)

    # 3. Correction Table (syndrome string 's3 s2 s1 s0')
    # This table IS for the 'XZZXI' stabilizers
    correction_table = {
        '0001': [('x', 0)],
        '0010': [('z', 2)],
        '0011': [('x', 4)],
        '0100': [('z', 4)],
        '0101': [('z', 1)],
        '0110': [('x', 3)],
        '0111': [('y', 4)],
        '1000': [('x', 1)],
        '1001': [('z', 3)],
        '1010': [('z', 0)],
        '1011': [('y', 0)],
        '1100': [('x', 2)],
        '1101': [('y', 1)],
        '1110': [('y', 2)],
        '1111': [('y', 3)]
    }

    # 4. Logical Z Measurement (Z_L = ZZZZZ)
    # This IS the correct measurer for this code.
    logical_z_measure = QuantumCircuit(qr_d, qr_lz, cr_l, name='measure_z_5q')
    logical_z_measure.h(qr_lz[0])
    for i in range(n_data):
        logical_z_measure.cz(qr_lz[0], qr_d[i])
    logical_z_measure.h(qr_lz[0])
    logical_z_measure.measure(qr_lz[0], cr_l[0])
    
    return {
        'name': '5-Qubit Code',
        'n_data': n_data,
        'n_ancilla': n_ancilla,
        'n_logical_z_anc': n_logical_z_anc,
        'encode_0': encode_0,
        'syndrome_measure': syndrome_measure,
        'correction_table': correction_table,
        'logical_z_measure': logical_z_measure
    }

# --- Simulation Infrastructure ---

def build_noise_model(p_phys):
    # 1. Gate noise: Applied after every gate
    gate_error_1 = depolarizing_error(p_phys, 1)
    gate_error_2 = depolarizing_error(p_phys, 2)
    
    # 2. "Idle" noise: Applied to our id gates
    
    # Use 'I' (capital-eye) for pauli_error
    idle_error = pauli_error([
        ('X', p_phys / 3),
        ('Y', p_phys / 3),
        ('Z', p_phys / 3),
        ('I', 1 - p_phys)
    ])
    noise_model = NoiseModel()
    
    # Add noise to all gates
    
    noise_model.add_all_qubit_quantum_error(gate_error_1, ['h', 's', 'z', 'x', 'y'])
    noise_model.add_all_qubit_quantum_error(gate_error_2, ['cx', 'cz', 'cy'])
   
    noise_model.add_all_qubit_quantum_error(idle_error, "id")
    
    return noise_model

def run_simulation(code_dict, p_phys, shots):
    """
    Runs a full QEC simulation (V19 logic).
    """
    
    # Get code properties
    n_data = code_dict['n_data']
    n_ancilla = code_dict['n_ancilla']
    n_logical_z_anc = code_dict['n_logical_z_anc']
    encode_0 = code_dict['encode_0']
    syndrome_measure = code_dict['syndrome_measure']
    correction_table = code_dict['correction_table']
    logical_z_measure = code_dict['logical_z_measure']

    # --- Build the full circuit ---
    qr_d = QuantumRegister(n_data, 'q')
    cr_l = ClassicalRegister(1, 'logical_out')
    
    all_q_regs = [qr_d]
    all_c_regs = [cr_l] # Logical bit is reg 0
    
    if n_ancilla > 0:
        qr_a = QuantumRegister(n_ancilla, 'a')
        cr_s = ClassicalRegister(n_ancilla, 'syn')
        all_q_regs.append(qr_a)
        all_c_regs.append(cr_s) # Syndrome is reg 1
        
    if n_logical_z_anc > 0:
        qr_lz = QuantumRegister(n_logical_z_anc, 'lz')
        all_q_regs.append(qr_lz)
        
    full_qc = QuantumCircuit(*all_q_regs, *all_c_regs)
    
    # 1. Encode |0>_L (from user's function)
    full_qc.compose(encode_0, qr_d, inplace=True)
    # The 'id' gate in encode_0 will have noise applied
    
    for i in range(15):
      full_qc.x(range(n_data))
      full_qc.x(range(n_data))

    # 2. Syndrome Measure (This will also have gate noise)
    if n_ancilla > 0:
        full_qc.compose(syndrome_measure, qr_d[:] + qr_a[:], cr_s, inplace=True)
        full_qc.barrier()

    # 3. Correction (Gate noise will be applied)
    if correction_table:
        for syndrome_str, ops in correction_table.items():
            syndrome_int = int(syndrome_str, 2)
            with full_qc.if_test((cr_s, syndrome_int)):
                for gate, qubit_idx in ops:
                    if gate == 'x':
                        full_qc.x(qr_d[qubit_idx])
                    elif gate == 'z':
                        full_qc.z(qr_d[qubit_idx])
                    elif gate == 'y':
                        full_qc.y(qr_d[qubit_idx])
        full_qc.barrier()

    # 4. Logical Z Measure (Gate noise will be applied)
    if n_logical_z_anc > 0:
        full_qc.compose(logical_z_measure, qr_d[:] + qr_lz[:], cr_l, inplace=True)
    else:
        # Baseline case
        full_qc.compose(logical_z_measure, qr_d, cr_l, inplace=True)
        
    # --- Run Simulation ---
    noise_model = build_noise_model(p_phys)
    sim = AerSimulator()
    t_qc = transpile(full_qc, sim,optimization_level=0)
    result = sim.run(t_qc, noise_model=noise_model, shots=shots).result()
    counts = result.get_counts(0)
    
    # --- Analyze results ---
    logical_errors = 0
    for output, count in counts.items():
        logical_bit_str = output.split(' ')[-1]
        if logical_bit_str == '1':
            logical_errors += count
            
    return logical_errors / shots

if __name__ == '__main__':
    
    # Simulation parameters
    SHOTS = 32768 
    PHYSICAL_ERROR_RATES = np.logspace(-4,-3, 10) # ~0.00003 to ~0.003
    
    # Get code definitions
    code_baseline = get_baseline_code()
    code_5q = get_5q_code()
    
    # Store results
    logical_error_rates_baseline = []
    logical_error_rates_5q = []
    
    print("Starting QEC simulations (V19, Final Syntax Fix)... This may take a while.")
    print(f"Running {SHOTS} shots for each of {len(PHYSICAL_ERROR_RATES)} physical error rates.")
    
    for p in PHYSICAL_ERROR_RATES:
        print(f"\nRunning for p_phys = {p:}")
        
        # Baseline
        print("  Simulating Baseline...")
        err_base = run_simulation(code_baseline, p, SHOTS)
        logical_error_rates_baseline.append(err_base)
        
        # 5-Qubit
        print("  Simulating 5-Qubit Code...")
        err_5q = run_simulation(code_5q, p, SHOTS)
        logical_error_rates_5q.append(err_5q)
        
        print(f"  Results:")
        print(f"    p_log (Baseline): {err_base:}")
        print(f"    p_log (5-Qubit):  {err_5q:}")

    # --- Visualization ---
    print("\nSimulation complete. Generating plot...")
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(PHYSICAL_ERROR_RATES, logical_error_rates_baseline, 'o-', label='Baseline (No QEC)', markersize=8)
    plt.plot(PHYSICAL_ERROR_RATES, logical_error_rates_5q, 's-', label='5-Qubit Code (User Encoder)', markersize=8)
    
    plt.plot(PHYSICAL_ERROR_RATES, PHYSICAL_ERROR_RATES, 'k--', label='Threshold (p_log = p_phys)')
    
    C = 10 
    p_squared = [C * (p**2) for p in PHYSICAL_ERROR_RATES]
    plt.plot(PHYSICAL_ERROR_RATES, p_squared, ':', color='grey', label=r'Theory guide ($p_{log} \propto p_{phys}^2$)')
    
    plt.xlabel('Physical Error Rate ($p_{phys}$)', fontsize=14)
    plt.ylabel('Logical Error Rate ($p_{log}$)', fontsize=14)
    plt.title('QEC Performance: 5-Qubit Code vs. Baseline (V19)', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_filename = 'qec_comparison_plot_v19.png'
    plt.savefig(output_filename)
    
    print(f"Plot saved as '{output_filename}'")
    
    plt.show()
