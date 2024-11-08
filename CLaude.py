import numpy as np
import matplotlib.pyplot as plt
import random

class HopfieldNetwork:
    def __init__(self, size=100):
        self.size = size
        self.weights = np.zeros((size, size))
        self.patterns = []
        
    def create_letter_patterns(self):
        """Create 10x10 binary patterns for letters A, C, K, T, W"""
        # Initialize patterns dictionary
        patterns = {}
        
        # Letter A pattern (10x10)
        patterns['A'] = np.array([
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
        ])
        
        # Letter C pattern
        patterns['C'] = np.array([
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
        ])
        
        # Letter K pattern
        patterns['K'] = np.array([
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
        ])
        
        # Letter T pattern
        patterns['T'] = np.array([
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
        ])
        
        # Letter W pattern
        patterns['W'] = np.array([
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
        ])
        
        # Convert 2D patterns to 1D vectors
        for key in patterns:
            self.patterns.append(patterns[key].flatten())
            
        return patterns
    
    def compute_weights(self):
        """Compute the weight matrix using outer product rule"""
        self.weights = np.zeros((self.size, self.size))
        
        # Sum outer products of all patterns
        for pattern in self.patterns:
            outer_product = np.outer(pattern, pattern)
            self.weights += outer_product
            
        # Set diagonal elements to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # Ensure symmetry
        self.weights = (self.weights + self.weights.T) / 2
        
        return self.weights
    
    def add_noise(self, pattern, std_dev):
        """Add Gaussian noise to a pattern and binarize"""
        noise = np.random.normal(0, std_dev, pattern.shape)
        noisy_pattern = pattern + noise
        # Binarize using sign function
        return np.sign(noisy_pattern)
    
    def update_neuron(self, state, index):
        """Update a single neuron using the Hopfield update rule"""
        h = np.dot(self.weights[index], state)
        return np.sign(h)
    
    def update_async(self, initial_state, max_epochs=100):
        """Asynchronous update of the network"""
        current_state = initial_state.copy()
        states_history = [current_state.copy()]
        
        for epoch in range(max_epochs):
            prev_state = current_state.copy()
            
            # Update neurons in random order
            update_order = list(range(self.size))
            random.shuffle(update_order)
            
            for i in update_order:
                current_state[i] = self.update_neuron(current_state, i)
            
            states_history.append(current_state.copy())
            
            # Check for convergence
            if np.array_equal(prev_state, current_state):
                print(f"Converged after {epoch + 1} epochs")
                break
                
        return current_state, states_history

    def plot_pattern(self, pattern, title="Pattern"):
        """Visualize a pattern as a 10x10 grid"""
        plt.figure(figsize=(4, 4))
        plt.imshow(pattern.reshape(10, 10), cmap='binary')
        plt.title(title)
        plt.axis('off')
        plt.show()
        

def main():
    # Initialize network and patterns once
    network = HopfieldNetwork(100)
    patterns = network.create_letter_patterns()
    network.compute_weights()
    
    # First, plot all stored patterns in one window
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for idx, (letter, pattern) in enumerate(patterns.items()):
        ax = axes[idx]
        ax.imshow(pattern, cmap='binary')
        ax.set_title(letter)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    def run_hopfield(letter, std_dev):
        print(f"\nTesting letter {letter} with noise std_dev = {std_dev}")
        original_pattern = patterns[letter].flatten()
        noisy_pattern = network.add_noise(original_pattern, std_dev)
        recovered_pattern, states_history = network.update_async(noisy_pattern)
        
        # Plot convergence process in one window
        num_plots = min(len(states_history), 20)
        step = max(1, len(states_history) // num_plots)
        selected_epochs = states_history[::step]
        
        cols = 6
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()
        
        for idx, epoch_pattern in enumerate(selected_epochs):
            ax = axes[idx]
            ax.imshow(epoch_pattern.reshape(10, 10), cmap='binary')
            ax.set_title(f"σ={std_dev}, Epoch {idx * step}")
            ax.axis('off')
        
        # Hide unused subplots
        for ax in axes[num_plots:]:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    # Run test cases
    run_hopfield("C", 1.1)
    run_hopfield("K", 0.5)
    run_hopfield("K", 0.8)

if __name__ == "__main__":
    main()