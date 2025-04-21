import cirq
import numpy as np
import sympy
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import OptimizeResult
from cma import CMAEvolutionStrategy
import torch
from torch.optim import Adam
import torch.nn as nn






#Classical Minimiser


def cmaes_to_adam_minimization(
    fun,
    bounds,
    args=(),
    num_points=10000,  # Number of points for Gaussian sampling
    sigma_param=0.1,  # Standard deviation for Gaussian sampling
    center_point=None,  # Point around which to sample
    cmaes_population_size=None,  # Auto-adjusted
    cmaes_max_iter=100,
    adam_lr=1e-5,
    adam_epochs=100,
    adam_batch_size=1,
):
    dim = len(bounds)
    lower_bounds, upper_bounds = np.array(bounds).T

    if cmaes_population_size is None:
        cmaes_population_size = 300 * dim  # Adaptive pop size

    # 1. GAUSSIAN SAMPLING INITIALIZATION
    if center_point is None:
        center_point = np.mean([lower_bounds, upper_bounds], axis=0)  # Default to center of bounds

    print("Starting Gaussian Sampling...")

    # Sample points using Gaussian distribution around the center_point
    sampled_points = np.random.normal(loc=center_point, scale=sigma_param, size=(num_points, dim))

    # Clip points to stay within bounds
    sampled_points = np.clip(sampled_points, lower_bounds, upper_bounds)

    # Evaluate function at sampled points and track the best found
    best_sample_x = None
    best_sample_val = float("inf")

    for i, x in enumerate(sampled_points):
        val = fun(x, *args)

        # Print progress every 1000 points or at the end
        if (i + 1) % 1000 == 0 or i == len(sampled_points) - 1:
            print(f"Sample {i + 1}/{num_points}")

        # Update best sample if a new best is found
        if val < best_sample_val:
            best_sample_val = val
            best_sample_x = x
            print(f"New Best Found: Value = {best_sample_val} at {best_sample_x}")

    print(f"Gaussian Sampling Completed. Best Initial Value: {best_sample_val} at {best_sample_x}")

    # 2. CMA-ES OPTIMIZATION
    sigma_init = np.mean(upper_bounds - lower_bounds) * 0.2  # Initial sigma for exploration
    cma_options = {
        "popsize": cmaes_population_size,
        "maxiter": cmaes_max_iter,
        "AdaptSigma": True,
        "tolupsigma": 1e5,  # Prevent sigma from shrinking too quickly
        "seed": None,  # Ensures variability in runs
    }

    print("Starting CMA-ES Optimization...")

    cma = CMAEvolutionStrategy(best_sample_x, sigma_init, cma_options)

    def objective_cmaes(x):
        x = np.clip(x, lower_bounds, upper_bounds)  # Enforce bounds
        return fun(x, *args)

    cma.optimize(objective_cmaes)
    cma_best_x = np.clip(cma.result.xbest, lower_bounds, upper_bounds)
    cma_best_val = fun(cma_best_x, *args)

    print(f"CMA-ES Completed. Best CMA-ES Value: {cma_best_val} at {cma_best_x}")

    # Ensure CMA-ES does not return a worse value than the sampled best
    if cma_best_val > best_sample_val:
        print("Warning: CMA-ES did not improve upon sampled best. Reverting to best sampled point.")
        cma_best_x = best_sample_x
        cma_best_val = best_sample_val

    # 3. ADAM GRADIENT REFINEMENT
    print("Starting ADAM Refinement...")

    class ParamModel(nn.Module):
        def __init__(self, initial_params):
            super().__init__()
            self.params = nn.Parameter(torch.tensor(initial_params, dtype=torch.float32, requires_grad=True))

        def forward(self):
            obj_value = fun(self.params.detach().numpy(), *args)
            return torch.tensor(obj_value, dtype=torch.float32, requires_grad=True)

    model = ParamModel(cma_best_x)
    optimizer = Adam([model.params], lr=adam_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_adam_val = float("inf")
    best_adam_x = None

    for epoch in range(adam_epochs):
        optimizer.zero_grad()
        obj_value = model()
        obj_value.backward()
        optimizer.step()

        # Clamp to stay within bounds
        with torch.no_grad():
            model.params.copy_(torch.clamp(model.params, min=torch.tensor(lower_bounds), max=torch.tensor(upper_bounds)))

        # Track the best solution found by ADAM
        current_val = obj_value.item()
        if current_val < best_adam_val:
            best_adam_val = current_val
            best_adam_x = model.params.detach().numpy()
            print(f"New Best Found by ADAM: Value = {best_adam_val} at {best_adam_x}")

        if epoch % 50 == 0 or epoch == adam_epochs - 1:
            print(f"Epoch {epoch}/{adam_epochs}: Objective Value = {obj_value.item()}")

    final_params = best_adam_x
    final_value = best_adam_val

    print(f"ADAM Completed. Best Final Value: {final_value} at {final_params}")
    
    return OptimizeResult(x=final_params, fun=final_value, success=True, message="Optimization complete")





#OOP to track variables after swapping

class LineCoefficientTracker:
    def __init__(self, qubits, qudit_dim, lat_dim):
        """
        Initializes the tracker for a flattened grid of qubits arranged as:
        [q0_0, q1_0, ..., q(qudit_dim-1)_0, q0_1, q1_1, ..., q(qudit_dim-1)_(lat_dim-1)]
        
        Parameters:
        - qubits: List of Cirq qubits.
        - qudit_dim: Number of qubits per lattice variable (qudit dimension).
        - lat_dim: Number of lattice variables (lattice dimension).
        """
        self.qudit_dim = qudit_dim
        self.lat_dim = lat_dim
        
        # Validate the total number of qubits
        assert len(qubits) == qudit_dim * lat_dim, "Number of qubits must be qudit_dim * lat_dim"
        
        # Initialize mappings for positions and qubits
        self.position_to_qubit = {
            (i, j): qubits[j * qudit_dim + i] for j in range(lat_dim) for i in range(qudit_dim)
        }
        self.qubit_to_position = {
            qubit: (i, j) for (i, j), qubit in self.position_to_qubit.items()
        }

    def apply_swap(self, q1, q2):
        """
        Swaps two qubits in the linear structure, updating their logical positions.
        
        Parameters:
        - q1, q2: The two Cirq qubits to swap.
        """
        # Get the logical positions of the qubits
        pos1 = self.qubit_to_position[q1]
        pos2 = self.qubit_to_position[q2]
        
        # Swap in position-to-qubit mapping
        self.position_to_qubit[pos1], self.position_to_qubit[pos2] = q2, q1
        
        # Swap in qubit-to-position mapping
        self.qubit_to_position[q1], self.qubit_to_position[q2] = pos2, pos1

    def get_original_position(self, qubit):
        """
        Returns the logical position (qudit_index, lattice_index) of the given qubit.
        
        Parameters:
        - qubit: The Cirq qubit.
        
        Returns:
        - (qudit_index, lattice_index): The original position of the qubit.
        """
        return self.qubit_to_position[qubit]

    def get_qubit_at_position(self, position):
        """
        Returns the qubit currently at the given logical position (qudit_index, lattice_index).
        
        Parameters:
        - position: A tuple (qudit_index, lattice_index).
        
        Returns:
        - The Cirq qubit at the specified position.
        """
        return self.position_to_qubit.get(position, None)







def build_qaoa_circuit(qudit_dim, lat_dim, n_qubits: int, p_layers: int, gram_matrix: np.ndarray, Z_dict, Z_Pairs, Z_single) -> (cirq.Circuit, list, list):
    """
    Builds a parameterized QAOA circuit with n qubits and p layers, with additional sequence steps
    in each layer to apply ZZ interactions and swaps as specified.

    Parameters:
    - n_qubits: Number of qubits in the line.
    - p_layers: Number of QAOA layers.
    - gram_matrix: A matrix of ZZ interaction coefficients.

    Returns:
    - A tuple containing the QAOA circuit, list of gamma symbols, and list of beta symbols.
    """
    qubits = cirq.LineQubit.range(n_qubits)
    qaoa_circuit = cirq.Circuit()
    
    tracker = LineCoefficientTracker(qubits, qudit_dim, lat_dim)

    # Initialize all qubits with Hadamard gates
    qaoa_circuit.append([cirq.H(q) for q in qubits])

    # Create symbols for gamma and beta parameters for each layer
    gammas = [sympy.Symbol(f"γ{layer+1}") for layer in range(p_layers)]
    betas = [sympy.Symbol(f"β{layer+1}") for layer in range(p_layers)]

    
    

    def apply_zz_with_gram(qaoa_circuit, gamma, gram_matrix, Z_dict, apply_to="all"):
        """
        Applies ZZ gates with coefficients from the Gram matrix based on logical positions,
        selectively applying to even or odd pairs if specified.
        """
        for i in range(n_qubits - 1):
            # Determine if the current index matches the specified `apply_to` condition
            if apply_to == "even" and i % 2 != 0:
                continue
            elif apply_to == "odd" and i % 2 == 0:
                continue
            
            
            original_position1 = tracker.get_original_position(qubits[i])
            original_position2 = tracker.get_original_position(qubits[i + 1])
            
            
            Z_coefficient = Z_dict.get((original_position1, original_position2))
            
            
            gram_coefficient = gram_matrix[original_position1[1], original_position2[1]]
            
            coefficient = Z_coefficient * gram_coefficient
            
            #x4 when lat index is different for terms which appear twice where QiQj = QjQi 
            if original_position1[1] != original_position2[1]:
                coefficient = coefficient * 2
                
            #Double for diagonal gram matrix coefficients
            if original_position1[1] == original_position2[1]:
                coefficient = coefficient * 2
            
            
            #Dont add any identity terms 
            if coefficient == 0:
                continue
            else:
                qaoa_circuit.append(cirq.ZZ(qubits[i], qubits[i + 1]) ** (gamma * coefficient))
                
                
            

    # Loop over QAOA layers
    for layer in range(p_layers):
        
        # For each layer, apply multiple sequences to get all ZZ terms
        for seq in range(n_qubits - 1):
            if seq == 0:
                # Sequence within Layer 0: Apply ZZ interactions between all consecutive qubits using Gram matrix
                apply_zz_with_gram(qaoa_circuit, gammas[layer], gram_matrix, Z_dict, apply_to="even")
                apply_zz_with_gram(qaoa_circuit, gammas[layer], gram_matrix, Z_dict, apply_to="odd")
            else:
                if seq % 2 == 1:
                    # Even layer (Python index odd): Swap even-indexed qubits with the right neighbor
                    for i in range(0, n_qubits - 1, 2):
                        qaoa_circuit.append(cirq.SWAP(qubits[i], qubits[i + 1]))
                        # Update logical positions after swap
                        tracker.apply_swap(qubits[i], qubits[i + 1])
                    
                    # Apply ZZ interactions between odd-indexed qubits
                    apply_zz_with_gram(qaoa_circuit, gammas[layer], gram_matrix, Z_dict, apply_to="odd")

                else:
                    # Odd layer (Python index even): Swap odd-indexed qubits with the right neighbor
                    for i in range(1, n_qubits - 1, 2):
                        qaoa_circuit.append(cirq.SWAP(qubits[i], qubits[i + 1]))
                        # Update logical positions after swap
                        tracker.apply_swap(qubits[i], qubits[i + 1])

                    # Apply ZZ interactions between even-indexed qubits
                    apply_zz_with_gram(qaoa_circuit, gammas[layer], gram_matrix, Z_dict, apply_to="even")


        #Apply Single qubit Z gate 2*lat_dim times
        for q in range(n_qubits):
            for iteration in range(lat_dim):
            
                
                original_position = tracker.get_original_position(qubits[q])
                Z_coefficient = Z_single[original_position[0]]
                
                #Get an array of possible indices not equal to the gram index
                indices = np.arange(lat_dim + 1)
                #filtered_array = indices[indices != original_position[1]]
                
                
                
                #Gram matrix is symmetric so doesnt matter ordering
                gram_coefficient = gram_matrix[original_position[1], indices[iteration]]
            
                coefficient = gram_coefficient * Z_coefficient
                
                #Account for single qubit gate twice as Gij = Gji
                coefficient = coefficient * 2
                
                
                if coefficient == 0:
                    continue
                else:
                    qaoa_circuit.append(cirq.Z(qubits[q]) ** (gammas[layer] * coefficient))
                    

        # Apply X rotations as driver for each qubit after ZZ interactions in each layer
        qaoa_circuit.append([cirq.X(q) ** betas[layer] for q in qubits])

    return qaoa_circuit, gammas, betas, tracker









def energy_from_wavefunction(wf: np.ndarray, qubits, gram_matrix: np.ndarray, tracker, Z_dict, Z_single, qudit_dim, lat_dim) -> float:
    """
    Computes energy based on ZZ interactions for all possible qubit pairs with weights from a Gram matrix.
    
    Parameters:
    - wf: Wavefunction (probability amplitude) array of the quantum state.
    - qubits: List of qubits in the system.
    - gram_matrix: Matrix containing the coefficients for ZZ interactions between qubit pairs.

    Returns:
    - A float representing the expected energy of the wavefunction.
    """
    n_qubits = len(qubits)
    
    # Construct the Z operator for each qubit basis state in the computational basis
    #Z = np.array([(-1) ** (np.arange(2 ** n_qubits) >> i & 1) for i in range(n_qubits)])
    Z = np.array([(-1) ** (np.arange(2 ** n_qubits) >> (n_qubits - 1 - i) & 1) for i in range(n_qubits)])
    
    # Build the weighted ZZ filter using the Gram matrix coefficients
    ZZ_filter = np.zeros(2 ** n_qubits)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            
            original_position1 = tracker.get_original_position(qubits[i])
            original_position2 = tracker.get_original_position(qubits[j])
            
            Z_coefficient = Z_dict.get((original_position1, original_position2))
            gram_coefficient = gram_matrix[original_position1[1], original_position2[1]]
            
            coefficient = Z_coefficient * gram_coefficient
            
            #x4 when lat index is different for terms which appear twice where QiQj = QjQi 
            if original_position1[1] != original_position2[1]:
                coefficient = coefficient * 2
                
            #Double for diagonal gram matrix coefficients
            if original_position1[1] == original_position2[1]:
                coefficient = coefficient * 2
            
            
            ZZ_filter += (Z[i] * Z[j] * coefficient)
    
    
    Z_filter = np.zeros(2 ** n_qubits)
    
    
    #Build the single qubit interaction energies
    for q in range(n_qubits):
        for iteration in range(lat_dim):
            
            original_position = tracker.get_original_position(qubits[q])
            Z_coefficient = Z_single[original_position[0]]
            
            #Get an array of possible indices that can produce single Z
            indices = np.arange(lat_dim + 1)
            #filtered_array = indices[indices != original_position[1]]
            
            #Gram matrix is symmetric so doesnt matter ordering
            gram_coefficient = gram_matrix[original_position[1], indices[iteration]]
        
            coefficient = gram_coefficient * Z_coefficient
            
            #Account for single qubit gate twice as Gij = Gji
            coefficient = coefficient * 2
            
            Z_filter += (Z[q] * coefficient) 
            
            
    
    
    
    #Adding constants for I/2 terms in the Hamiltonian
    const1 = 0
    for i in range(len(gram_matrix)):
        for j in range(len(gram_matrix)):
            const1 += gram_matrix[i, j] * 0.25
            
            
    #Add constant terms for diagonal terms where same Z operator is squared
    const2 = 0
    for i in range(len(gram_matrix)):
        for j in range(len(Z_single)):
            x = Z_single[j] * 2 #Undo factor of 1/2 to get Z_base 
            y = x**2
            const2 += gram_matrix[i, i] * y
            
            
    const = const1 + const2
    
    const_array = np.full(2 ** n_qubits, const)
    
    
    
    # Compute the energy by taking the weighted sum over the probability distribution from the wavefunction
    probabilities = np.abs(wf) ** 2
    energy_eigenvalues = ZZ_filter + Z_filter + const_array
    
    energy = np.sum(probabilities * (ZZ_filter + Z_filter + const_array))
    
    
    
    # Find the most probable computational basis state and its energy
    max_index = np.argmax(probabilities)
    max_state = f"{max_index:0{n_qubits}b}"  # Binary string representation
    
    most_probable_state = np.argmax(probabilities)
    energy_most_probable = (energy_eigenvalues)[most_probable_state]
    # print("ENEGRY MOST PROBABLE:", energy_most_probable)
    

    
    w = 100
    
    
    return energy + (w*energy_most_probable)






def energy_from_params(gamma_values: list, beta_values: list, qaoa: cirq.Circuit, qubits, gram_matrix, tracker, Z_dict, Z_single, qudit_dim, lat_dim) -> float:
    """Computes the energy given gamma and beta parameter values."""
    sim = cirq.Simulator()
    # Create a parameter resolver with dynamically generated gamma and beta values
    params = cirq.ParamResolver({
        **{f"γ{idx+1}": gamma_values[idx] for idx in range(len(gamma_values))},
        **{f"β{idx+1}": beta_values[idx] for idx in range(len(beta_values))}
    })
    result = sim.simulate(qaoa, param_resolver=params)
    wf = result.final_state_vector
    return energy_from_wavefunction(wf, qubits, gram_matrix, tracker, Z_dict, Z_single, qudit_dim, lat_dim)



def objective_function(params, qaoa, qubits, p_layers, gram_matrix, tracker, Z_dict, Z_single, qudit_dim, lat_dim):
    """Objective function for optimization: energy as a function of parameters."""
    
    if isinstance(params, torch.Tensor):
        # For PyTorch, detach from computation graph and convert to NumPy array
        params = params.detach().numpy()
    
    gamma_values = params[:p_layers]
    beta_values = params[p_layers:]
    return energy_from_params(gamma_values, beta_values, qaoa, qubits, gram_matrix, tracker, Z_dict, Z_single, qudit_dim, lat_dim)



def main():
    # Define QAOA parameters
    p_layers = 2  # Number of QAOA layers 
    
    lat_dim = 3
    qudit_dim = 2
    
    n_qubits = lat_dim * qudit_dim
    
    
    
    #Function to compute Gram matrix for some basis vectors
    def Gram(lat_dim):
       
        x1 = np.array([5, 13, 19])
        x2 = np.array([7, 10, 16])
        x3 = np.array([14, 8, 3])
        
        
        
        basis_vectors = [x1, x2, x3]
        
        norms = []
        for x in basis_vectors:
            norms.append(np.sum(x**2))
            
            
            
        
        B = np.stack(basis_vectors, axis = 1)
        
        G = B.T @ B
        
        return G, norms
    
    gram_matrix, norms = Gram(lat_dim)
    
    
    
    
    #Function to compute coefficients for specified dimensions
    def Z_coeff(qudit_dim, lat_dim):
        # Step 1: Generate the base Z_coeff array with values 2^(i-1)
        Z_base = np.array([2 ** (i - 1) for i in range(qudit_dim)])

        # Step 2: Create `lat_dim` copies of Z_coeff and label each element with its [i, j] index
        lattices = []
        for j in range(lat_dim):
            lattice = [(value, i, j) for i, value in enumerate(Z_base)]
            lattices.append(lattice)

        # Flatten `lattices` to get a single list with all elements and their labels
        flattened_lattice = [item for lattice in lattices for item in lattice]

        # Step 3: Generate all possible products including self-interacting terms
        Z_pairs = []
        for item1, item2 in itertools.product(flattened_lattice, repeat=2):
            value1, i1, j1 = item1
            value2, i2, j2 = item2
            product = value1 * value2
            Z_pairs.append(((i1, j1), (i2, j2), product))

        # Step 4: Generate the Z_single array with each element scaled by 0.5
        Z_single = Z_base * 0.5

        return Z_pairs, Z_single

    
    
    #Get the Z coefficients
    Z_pairs, Z_single = Z_coeff(qudit_dim, lat_dim)
    

    #Change Z_pairs to a directory to more easily call values
    Z_dict = {((i1, j1), (i2, j2)): product for (i1, j1), (i2, j2), product in Z_pairs}
    print(Z_dict)
    

    # Build QAOA circuit
    qaoa_circuit, gammas, betas, tracker = build_qaoa_circuit(qudit_dim, lat_dim, n_qubits, p_layers, gram_matrix, Z_dict, Z_pairs, Z_single)
    qubits = cirq.LineQubit.range(n_qubits)


    print(f"Generated QAOA Circuit for {n_qubits} qubits and {p_layers} layers:")
    print(qaoa_circuit)
    print("\n" + "="*50 + "\n")

    
    
    
    bounds = []
    for i in range(p_layers):
        bounds.append((0, 1))
        bounds.append((0, 1))
    
    
    result = cmaes_to_adam_minimization(
    objective_function,
    bounds=bounds,
    args=(qaoa_circuit, qubits, p_layers, gram_matrix, tracker, Z_dict, Z_single, qudit_dim, lat_dim))
    
    
    print("Optimized parameters (gamma1, gamma2, ..., beta1, beta2, ...):", result.x)
    print("Minimum energy:", result.fun)
    
    
    
    
    
    #Grid Search for p=1 with one gamma and one beta
    if p_layers == 1:
        """Do a grid search over values of 𝛄 and β."""
        # Set the grid size and range of parameters.
        grid_size = 100
        gamma_max = 2
        beta_max = 2
        
        
        
        # Do the grid search.
        energies = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                params = [i * gamma_max / grid_size, j * beta_max / grid_size]
                energies[i, j] = objective_function(params, qaoa_circuit, qubits, p_layers, gram_matrix, tracker, Z_dict, Z_single, qudit_dim, lat_dim)
        
        
        """Plot the energy as a function of the parameters 𝛄 and β found in the grid search."""
        plt.ylabel(r"$\frac{\gamma}{\pi}$")
        plt.xlabel(r"$\frac{\beta}{\pi}$")
        plt.title("Energy Values - Standard QAOA")
        plt.imshow(energies, extent=(0, beta_max, gamma_max, 0))
        plt.colorbar()
        plt.show()
    
    

    # === Wavefunction-based evaluation of QAOA output ===
    
    def compute_energy(meas: np.ndarray) -> float:
        """
        Computes the energy for a given measurement based on ZZ interactions.
        
        Parameters:
        - meas: A binary array (0s and 1s) representing a bitstring measurement outcome.
    
        Returns:
        - Energy value for the given measurement.
        """
        Z_vals = 1 - 2 * meas  # Map 0 -> 1 and 1 -> -1 for Pauli Z values
        #energy = sum(-Z_vals[i] * Z_vals[j] * gram_matrix[i, j] for i in range(n_qubits) for j in range(i + 1, n_qubits))
        
        
        energy = 0
        
        #Double qubit energy
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                
                original_position1 = tracker.get_original_position(qubits[i])
                original_position2 = tracker.get_original_position(qubits[j])
                
                Z_coefficient = Z_dict.get((original_position1, original_position2))
                gram_coefficient = gram_matrix[original_position1[1], original_position2[1]]
                
                coefficient = Z_coefficient * gram_coefficient
                
                
                #x4 when lat index is different for terms which appear twice where QiQj = QjQi 
                if original_position1[1] != original_position2[1]:
                    coefficient = coefficient * 2
                    
                #Double for diagonal gram matrix coefficients
                if original_position1[1] == original_position2[1]:
                    coefficient = coefficient * 2
            
            
                
                energy += Z_vals[i] * Z_vals[j] * coefficient
                
        
        #Single qubit energy
        for q in range(n_qubits):
            for iteration in range(lat_dim):
                
                original_position = tracker.get_original_position(qubits[q])
                Z_coefficient = Z_single[original_position[0]]
                
                #Get an array of possible indices that can produce single Z interactions
                indices = np.arange(lat_dim + 1)
                #filtered_array = indices[indices != original_position[1]]
                
                #Gram matrix is symmetric so doesnt matter ordering
                gram_coefficient = gram_matrix[original_position[1], indices[iteration]]
            
                coefficient = gram_coefficient * Z_coefficient
                
                #Account for single qubit gate twice as Gij = Gji
                coefficient = coefficient * 2
                
                energy += (Z_vals[q] * coefficient) 

        
        #Add constant terms for I/2 in Hamiltonian
        for i in range(lat_dim):
            for j in range(lat_dim):
                energy += gram_matrix[i, j] * 0.25
                
        #Add constant terms for diagonal terms where same Z operator is squared
        for i in range(len(gram_matrix)):
            for j in range(len(Z_single)):
                x = Z_single[j] * 2 #Undo factor of 1/2 to get Z_base 
                y = x**2
                energy += gram_matrix[i, i] * y
        
        
        return energy
    
    
    
    # Clone QAOA circuit WITHOUT measurement
    wavefunction_circuit = qaoa_circuit.copy()
    
    # Optimized parameter resolver
    gamma_opt = result.x[:p_layers]
    beta_opt = result.x[p_layers:]
    params = cirq.ParamResolver({
        **{f"γ{idx+1}": gamma_opt[idx] for idx in range(p_layers)},
        **{f"β{idx+1}": beta_opt[idx] for idx in range(p_layers)}
    })
    
    
    
    
    # Run simulation to get wavefunction
    simulator = cirq.Simulator()
    sim_result = simulator.simulate(wavefunction_circuit, param_resolver=params)
    
    # Get state vector and compute exact probabilities
    state_vector = sim_result.final_state_vector
    probs = np.abs(state_vector) ** 2
    bitstrings = [format(i, f'0{n_qubits}b') for i in range(len(probs))]
    
    # Sort most probable bitstrings
    num = 10  # Number of most common bitstrings to display
    sorted_indices = np.argsort(probs)[::-1]
    configs = [int(bitstrings[i], 2) for i in sorted_indices[:num]]  # integer indices for plotting
    probs_top = [probs[i] for i in sorted_indices[:num]]
    
    # Plot probabilities of the most common bitstrings
    plt.figure(figsize=(10, 6))
    plt.title(f"Probability of {num} Most Common Outputs")
    plt.bar(range(len(probs_top)), probs_top, tick_label=[f"{c:0{n_qubits}b}" for c in configs])
    plt.xlabel("Bitstring Configuration")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.show()
    
    # Compute energy of most probable configurations
    meas = [[int(bit) for bit in f"{c:0{n_qubits}b}"] for c in configs]
    costs = [compute_energy(np.array(m)) for m in meas]
    
    # Plot energies
    plt.figure(figsize=(10, 6))
    plt.title(f"Energy of {num} Most Common Outputs")
    plt.bar(range(len(costs)), costs, tick_label=[f"{c:0{n_qubits}b}" for c in configs])
    plt.xlabel("Bitstring Configuration")
    plt.ylabel("Energy")
    plt.xticks(rotation=45)
    plt.show()
    
    print(f"Fraction of outputs displayed: {np.sum(probs_top).round(2)}")
    
    # Compute energy histogram over *all* outcomes
    from collections import defaultdict
    
    energy_prob = defaultdict(float)
    
    for i, p in enumerate(probs):
        bitstring = format(i, f'0{n_qubits}b')
        meas = np.array([int(bit) for bit in bitstring])
        energy = compute_energy(meas)
        energy_prob[energy] += p
    
    # Plot energy distribution
    sorted_energies = sorted(energy_prob.keys())
    sorted_probs = [energy_prob[e] for e in sorted_energies]
    
    # Compute probability of energy = 0
    prob_energy_zero = energy_prob.get(0, 0) * 100  # as percent
    print("Probability of energy = 0:", prob_energy_zero)
    
    plt.figure(figsize=(16, 6))
    plt.title("Standard QAOA", fontsize=20, fontname='Times New Roman', fontweight='bold')
    plt.bar(range(len(sorted_energies)), sorted_probs, tick_label=sorted_energies, edgecolor='black', zorder=5)
    plt.xlabel("Energy Eigenvalue", fontsize=18, fontname='Times New Roman')
    plt.ylabel("Probability", fontsize=18, fontname='Times New Roman')
    plt.xticks(rotation=45, fontsize=14, fontname='Times New Roman')
    plt.yticks(fontsize=14, fontname='Times New Roman')
    plt.show()


    

    


if __name__ == "__main__":
    main()