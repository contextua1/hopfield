import numpy as np
import matplotlib.pyplot as plt
import random

# define 10x10 grids for letters A,C,K,T,W using -1 and 1
# implement using a dictionary where keys are uppercase letter strings and values are 2d lists
letters = {
    "A": [
            [-1,-1,-1,1,1,1,1,-1,-1,-1],
            [-1,-1,1,1,-1,-1,1,1,-1,-1],
            [-1,1,1,-1,-1,-1,-1,1,1,-1],
            [-1,1,1,-1,-1,-1,-1,1,1,-1],
            [-1,1,1,1,1,1,1,1,1,-1],
            [1,1,1,-1,-1,-1,-1,1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1]
    ],
    "C": [
            [-1,-1,1,1,1,1,1,1,-1,-1],
            [-1,1,1,-1,-1,-1,-1,1,1,-1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,-1,-1],
            [1,1,-1,-1,-1,-1,-1,-1,-1,-1],
            [1,1,-1,-1,-1,-1,-1,-1,-1,-1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [-1,1,1,-1,-1,-1,-1,1,1,-1],
            [-1,-1,1,1,1,1,1,1,-1,-1]
    ],
    "K": [
            [1,1,-1,-1,-1,-1,1,1,1,-1],
            [1,1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,-1,-1,1,1,1,-1,-1,-1],
            [1,1,-1,1,1,1,-1,-1,-1,-1],
            [1,1,1,1,1,-1,-1,-1,-1,-1],
            [1,1,1,1,1,-1,-1,-1,-1,-1],
            [1,1,-1,-1,1,1,1,-1,-1,-1],
            [1,1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,-1,-1,-1,-1,1,1,1,-1],
            [1,1,-1,-1,-1,-1,-1,1,1,1]
    ],
    "T": [
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,-1]
    ],
    "W": [
        [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,1,1,-1,-1,1,1],
            [1,1,-1,1,1,1,1,-1,1,1],
            [1,1,1,1,-1,-1,1,1,1,1],
            [1,1,1,-1,-1,-1,-1,1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1],
            [1,1,-1,-1,-1,-1,-1,-1,1,1]
    ]
}

# plot the letters A, C, K, T, W using matplotlib
fig, axes = plt.subplots(1, 5, figsize=(15, 5), facecolor='black')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
for idx, (letter, grid) in enumerate(letters.items()):
    ax = axes[idx]
    ax.imshow(grid, cmap="gray")
    ax.set_title(letter, color='white')
    ax.axis("off")
    ax.set_facecolor('black')
plt.tight_layout()
plt.show()

# convert each grid to a flattened numpy vector using dictionary comprehension
# constructs a new dictionary with the same keys (letters) but values are 100x1 numpy arrays
letter_vectors = {letter: np.array(grid).flatten() for letter, grid in letters.items()}
stored_patterns = list(letter_vectors.values())

#performs the weight matrix calculation
def create_weight_matrix(patterns):
    size = patterns[0].size
    W = np.zeros((size, size))
    for pattern in patterns:
        W += np.outer(pattern, pattern)
    np.fill_diagonal(W, 0)
    return (W + W.T) / 2 #ensures symmetry
#creates the weight matrix
Wm = create_weight_matrix(stored_patterns)


# add gaussian noise to the patterns
def add_noise(vector, sigma):
    # generate noise: ε ~ N(0, σ^2)
    noise = np.random.normal(0, sigma, vector.shape) #vector.shape means the noise is the same shape as the vector
    noisy_pattern = vector + noise #add noise to the vector
    # v_i = sign(v_i + ε_i) where v_i ∈ {-1, 1}
    return np.sign(noisy_pattern) #binarize the noisy pattern using the sign function

# asynchronous update function for hopfield network (one neuron update per iteration, visualized by epoch)
def hopfield_update_async(pattern, max_epochs=100):
    current_pattern = pattern.copy() #copy the pattern into a new variable
    history = [current_pattern.copy()] #initialize the epoch starting with epoch0 i.e noisy pattern
    num_neurons = len(current_pattern) #define the number of neurons as the length of the current pattern

    for epoch in range(max_epochs): 
        for _ in range(num_neurons): 
            # randomly select one neuron to update: i ∈ {0, 1, ..., 99}
            i = random.randint(0, num_neurons - 1) #randomly select a neuron
            update = np.sign(np.dot(Wm[i], current_pattern)) #update the neuron using the weight matrix and the current pattern
            current_pattern[i] = update if update != 0 else current_pattern[i] #update the current pattern with the new value
        history.append(current_pattern.copy()) #append the current pattern to the history list after each neuron update
        # check for convergence: if current pattern is same as previous
        if any(np.array_equal(current_pattern, stored_pattern) for stored_pattern in stored_patterns):
            break  # true convergence achieved (the current pattern matches one of the stored patterns)
    return history 

# run and plot the hopfield network for a specific letter and sigma value
def run_hopfield(letter="A", sigma=0.8):
    # retrieve state vector and weight matrix for specified letter
    vector = letter_vectors[letter]  #vector becomes the state vector for the argument letter
    
    # add gaussian noise to original pattern
    noisy_pattern = add_noise(vector, sigma)
    
    # run Hopfield network asynchronously
    history = hopfield_update_async(noisy_pattern)
    
    # plot selected epochs to show gradual convergence
    num_plots = min(len(history), 20)  # maximum of 20 plots for readability
    step = max(1, len(history) // num_plots)
    selected_epochs = history[::step] #if there are too many epochs, select only a few of them

    # dynamically determine number of rows and columns based on num_plots
    cols = 6
    rows = (num_plots + cols - 1) // cols  # ceiling division for number of rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), facecolor='black')
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    axes = axes.flatten()  # flatten in case of multiple rows
    for idx, epoch_pattern in enumerate(selected_epochs):
        ax = axes[idx]
        ax.imshow(epoch_pattern.reshape(10, 10), cmap="gray")
        ax.set_title(f"Sigma: {sigma}, Epoch {idx * step}", color='white')
        ax.axis("off")
        ax.set_facecolor('black')
    # hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis("off")
        ax.set_facecolor('black')
    plt.tight_layout()
    plt.show()

#usage
if __name__ == "__main__":
    run_hopfield("A", 55555.5)
    """run_hopfield("A", 0.8)
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
    """