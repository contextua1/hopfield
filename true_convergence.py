import numpy as np
import matplotlib.pyplot as plt
import random

# define 10x10 grids for letters A,C,K,T,W using -1 and 1
# implement using a dictionary where keys are uppercase letter strings and values are 2d lists
letters = {
    "A": [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
        [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    "C": [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    "K": [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1,  1, -1, -1],
        [-1,  1, -1, -1, -1,  1,  1, -1, -1, -1],
        [-1,  1, -1, -1,  1, -1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1,  1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1,  1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1,  1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    "T": [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ],
    "W": [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
        [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
        [-1, -1,  1, -1,  1,  1, -1,  1, -1, -1],
        [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
        [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
}

# plot the letters A, C, K, T, W using matplotlib
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for idx, (letter, grid) in enumerate(letters.items()):
    ax = axes[idx]
    ax.imshow(grid, cmap="gray")
    ax.set_title(letter)
    ax.axis("off")

plt.tight_layout()
plt.show()

# convert each grid to a flattened numpy vector using dictionary comprehension
# constructs a new dictionary with the same keys (letters) but values are 100x1 numpy arrays
letter_vectors = {letter: np.array(grid).flatten() for letter, grid in letters.items()}
stored_patterns = list(letter_vectors.values())
# define a function to compute weight matrix for a given vector
# weight matrix w = v * v^t - i, where:
# v is the state vector (100x1)
# v^t is the transpose of v (1x100)
# I is the identity matrix (100x100)
def compute_weight_matrix(vector):
    return np.outer(vector, vector) - np.eye(len(vector))

# compute weight matrices for each letter
# create a new dictionary with same keys but 100x100 weight matrix values
weight_matrices = {letter: compute_weight_matrix(vector) for letter, vector in letter_vectors.items()}

# add gaussian noise to the patterns
def add_noise(vector, sigma):
    """
    add gaussian noise to the input vector and binarize the result.

    parameters:
    - vector (np.ndarray): original state vector
    - sigma (float): standard deviation of the gaussian noise

    returns:
    - np.ndarray: noisy pattern binarized to {-1, 1}
    """
    # generate noise: ε ~ N(0, σ^2)
    noise = np.random.normal(0, sigma, vector.shape) #vector.shape means the noise is the same shape as the vector
    noisy_pattern = vector + noise #add noise to the vector
    # binarize the noisy pattern using the sign function
    # v_i = sign(v_i + ε_i) where v_i ∈ {-1, 1}
    noisy_pattern_binary = np.sign(noisy_pattern) #binarize the noisy pattern using the sign function
    return noisy_pattern_binary

# asynchronous update function for hopfield network (one neuron update per iteration, visualized by epoch)
def hopfield_update_async(weight_matrix, pattern, max_epochs=100):
    """
    update the network asynchronously until convergence or maximum epochs reached.

    parameters:
    - weight_matrix (np.ndarray): weight matrix W (100x100)
    - pattern (np.ndarray): initial state vector (100x1)
    - max_epochs (int): maximum number of epochs to run

    returns:
    - list of np.ndarray: history of state vectors over epochs
    """
    current_pattern = pattern.copy() #copy the pattern into a new variable
    history = [current_pattern.copy()] #initialize the epoch starting with epoch0 i.e noisy pattern
    num_neurons = len(current_pattern) #define the number of neurons as the length of the current pattern

    for epoch in range(max_epochs): #loop through the maximum number of epochs
        for _ in range(num_neurons): 
            # randomly select one neuron to update: i ∈ {0, 1, ..., 99}
            i = random.randint(0, num_neurons - 1) #randomly select a neuron
            # update rule: v_i(t+1) = sign(Σ_j W_ij * v_j(t))
            update = np.sign(np.dot(weight_matrix[i], current_pattern)) #update the neuron using the weight matrix and the current pattern
            current_pattern[i] = update #update the current pattern with the new value
        history.append(current_pattern.copy()) #append the current pattern to the history list after each neuron update
        # check for convergence: if current pattern is same as previous
        if any(np.array_equal(current_pattern, stored_pattern) for stored_pattern in stored_patterns):
            break  # true convergence achieved (the current pattern matches one of the stored patterns)

    return history 

# run and plot the hopfield network for a specific letter and sigma value
def run_hopfield(letter="A", sigma=0.8): #default letter is A and sigma is 0.8
    """
    run Hopfield network for specified letter with added noise and plot convergence process.

    parameters:
    - letter (str): target letter to retrieve
    - sigma (float): standard deviation of Gaussian noise added to initial pattern
    """
    # retrieve state vector and weight matrix for specified letter
    vector = letter_vectors[letter]  #vector becomes the state vector for the argument letter
    weight_matrix = weight_matrices[letter] #weight_matrix becomes the weight matrix for the argument letter
    
    # add gaussian noise to original pattern
    noisy_pattern = add_noise(vector, sigma)
    
    # run Hopfield network asynchronously
    history = hopfield_update_async(weight_matrix, noisy_pattern)
    
    # plot selected epochs to show gradual convergence
    num_plots = min(len(history), 20)  # maximum of 20 plots for readability
    step = max(1, len(history) // num_plots)
    selected_epochs = history[::step] 

    # dynamically determine number of rows and columns based on num_plots
    cols = 6
    rows = (num_plots + cols - 1) // cols  # ceiling division for number of rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # adjust figure size based on grid
    axes = axes.flatten()  # flatten in case of multiple rows
    for idx, epoch_pattern in enumerate(selected_epochs):
        ax = axes[idx]
        # reshape flat vector back to 10x10 grid for visualization
        ax.imshow(epoch_pattern.reshape(10, 10), cmap="gray")
        ax.set_title(f"Sigma: {sigma}, Epoch {idx * step}")
        ax.axis("off")
    
    # hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()

# example usage
if __name__ == "__main__":
    run_hopfield("A", 0.5)
    run_hopfield("A", 0.8)
    run_hopfield("A", 1.1)
    run_hopfield("C", 0.5)
    run_hopfield("C", 0.8)
    run_hopfield("C", 1.1)
    run_hopfield("K", 0.5)
    run_hopfield("K", 0.8)
    run_hopfield("K", 1.1)
    run_hopfield("T", 0.5)
    run_hopfield("T", 0.8)
    run_hopfield("T", 1.1)
    run_hopfield("W", 0.5)
    run_hopfield("W", 0.8)
    run_hopfield("W", 1.1)
