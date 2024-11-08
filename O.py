import numpy as np
import matplotlib.pyplot as plt
import random

def create_letter_A():
    A = np.ones((10,10)) * -1
    # Left side of A
    for i in range(1,9):
        A[i, 2] = 1
    # Right side of A
    for i in range(1,9):
        A[i, 7] = 1
    # Middle bar of A
    for j in range(2,8):
        A[5, j] = 1
    # Top of A
    for j in range(3,7):
        A[1, j] = 1
    A[2,2] = 1
    A[2,7] = 1
    return A

def create_letter_C():
    C = np.ones((10,10)) * -1
    # Top bar
    for j in range(2,8):
        C[1,j] = 1
    # Bottom bar
    for j in range(2,8):
        C[8,j] = 1
    # Left side
    for i in range(2,8):
        C[i,1] = 1
    return C

def create_letter_K():
    K = np.ones((10,10)) * -1
    # Left side of K
    for i in range(1,9):
        K[i,2] = 1
    # Diagonal lines
    for i in range(1,5):
        K[i, 7 - i + 1] = 1
    for i in range(5,9):
        K[i, i - 3] = 1
    return K

def create_letter_T():
    T = np.ones((10,10)) * -1
    # Top bar
    for j in range(1,9):
        T[1,j] = 1
    # Middle column
    for i in range(2,9):
        T[i,4] = 1
        T[i,5] = 1
    return T

def create_letter_W():
    W = np.ones((10,10)) * -1
    # Left side of W
    for i in range(1,9):
        W[i,1] = 1
    # Right side of W
    for i in range(1,9):
        W[i,8] = 1
    # Middle lines
    for i in range(5,9):
        W[i, i - 3] = 1
        W[i, 11 - i] = 1
    return W

def display_letter(letter_array, title):
    plt.imshow(letter_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def add_noise(pattern, sigma):
    noise = np.random.normal(0, sigma, pattern.shape)
    noisy_pattern = pattern + noise
    noisy_pattern = np.sign(noisy_pattern)
    # Replace zeros with 1 (since sign(0) = 0)
    noisy_pattern[noisy_pattern == 0] = 1
    return noisy_pattern

def hopfield_update_async(pattern, W, max_epochs=10):
    current_pattern = pattern.copy()
    num_neurons = len(pattern)
    for epoch in range(max_epochs):
        prev_pattern = current_pattern.copy()
        # Random order of neuron updates
        neuron_indices = list(range(num_neurons))
        random.shuffle(neuron_indices)
        for i in neuron_indices:
            # Update neuron i
            h = np.dot(W[i], current_pattern)
            current_pattern[i] = np.sign(h)
            if current_pattern[i] == 0:
                current_pattern[i] = 1
        # Visualize the pattern
        plt.imshow(current_pattern.reshape(10,10), cmap='gray')
        plt.title(f'Epoch {epoch+1}')
        plt.axis('off')
        plt.show()
        # Check for convergence
        if np.array_equal(current_pattern, prev_pattern):
            print(f'Converged after {epoch+1} epochs')
            break
    return current_pattern

def main():
    # Create the letters
    A = create_letter_A()
    C = create_letter_C()
    K = create_letter_K()
    T = create_letter_T()
    W_letter = create_letter_W()

    # Flatten the letters into vectors
    A_vec = A.flatten()
    C_vec = C.flatten()
    K_vec = K.flatten()
    T_vec = T.flatten()
    W_vec = W_letter.flatten()

    # Create patterns array
    patterns = np.array([A_vec, C_vec, K_vec, T_vec, W_vec])
    num_neurons = patterns.shape[1]

    # Compute the weight matrix
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        W += np.outer(p, p)
    # Set diagonal elements to zero
    np.fill_diagonal(W, 0)
    # Ensure symmetry
    W = (W + W.T) / 2

    # Verify the weight matrix is symmetric and diagonal is zero
    assert np.allclose(W, W.T), "Weight matrix W is not symmetric."
    assert np.all(np.diag(W) == 0), "Diagonal elements of W are not zero."

    # Display the letters
    letters = [A, C, K, T, W_letter]
    letter_names = ['A', 'C', 'K', 'T', 'W']

    # Map letter names to their vector representations
    letter_vectors = {'A': A_vec, 'C': C_vec, 'K': K_vec, 'T': T_vec, 'W': W_vec}

    for letter_array, name in zip(letters, letter_names):
        display_letter(letter_array, f'Letter {name}')

    # Add noise to each pattern and test
    sigma_values = [0.5, 0.8, 1.1]

    for letter_name in letter_names:
        original_pattern = letter_vectors[letter_name]
        print(f'Processing letter {letter_name}')
        for sigma in sigma_values:
            print(f'Adding noise with sigma = {sigma}')
            noisy_pattern = add_noise(original_pattern, sigma)
            # Display the noisy pattern
            plt.imshow(noisy_pattern.reshape(10,10), cmap='gray')
            plt.title(f'Noisy {letter_name} (sigma={sigma})')
            plt.axis('off')
            plt.show()
            # Use the Hopfield network to retrieve the pattern
            retrieved_pattern = hopfield_update_async(noisy_pattern, W, max_epochs=10)
            # Display the retrieved pattern
            plt.imshow(retrieved_pattern.reshape(10,10), cmap='gray')
            plt.title(f'Retrieved {letter_name} (sigma={sigma})')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    main()
