import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import os

# --- 0. Custom Particle Swarm Optimizer (PSO) Class ---
# We implement this from scratch to avoid library dependency errors
# and to show you exactly how the math works.
class CustomPSO:
    def __init__(self, n_particles, dimensions, options, bounds):
        self.n_particles = n_particles
        self.dims = dimensions
        self.c1 = options['c1'] # Cognitive parameter (follow own best)
        self.c2 = options['c2'] # Social parameter (follow swarm best)
        self.w = options['w']   # Inertia weight
        self.bounds = bounds
        
        # Initialize particle positions and velocities
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (n_particles, dimensions))
        self.velocities = np.zeros((n_particles, dimensions))
        
        # Best known positions
        self.pbest_pos = self.positions.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        
        # Global best
        self.gbest_pos = np.zeros(dimensions)
        self.gbest_cost = np.inf

    def optimize(self, cost_func, iters):
        print(f"Starting PSO Optimization ({iters} iterations)...")
        
        for i in range(iters):
            # 1. Evaluate fitness for all particles
            costs = cost_func(self.positions)
            
            # 2. Update Personal Bests
            better_mask = costs < self.pbest_cost
            self.pbest_pos[better_mask] = self.positions[better_mask]
            self.pbest_cost[better_mask] = costs[better_mask]
            
            # 3. Update Global Best
            min_cost_idx = np.argmin(self.pbest_cost)
            if self.pbest_cost[min_cost_idx] < self.gbest_cost:
                self.gbest_cost = self.pbest_cost[min_cost_idx]
                self.gbest_pos = self.pbest_pos[min_cost_idx].copy()
                print(f"  Iter {i+1}/{iters}: New Best MSE = {self.gbest_cost:.5f}")
            
            # 4. Update Velocities and Positions
            r1 = np.random.rand(self.n_particles, self.dims)
            r2 = np.random.rand(self.n_particles, self.dims)
            
            # Velocity update formula: Inertia + Cognitive + Social
            self.velocities = (self.w * self.velocities) + \
                              (self.c1 * r1 * (self.pbest_pos - self.positions)) + \
                              (self.c2 * r2 * (self.gbest_pos - self.positions))
            
            # Position update
            self.positions = self.positions + self.velocities
            
            # Clip positions to bounds (keep weights reasonable)
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
            
        return self.gbest_cost, self.gbest_pos

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")
if not os.path.exists('fso_dataset.csv'):
    print("\nCRITICAL ERROR: 'fso_dataset.csv' not found!")
    print("1. Open fso-simulator.html in your browser.")
    print("2. Use the 'Data Generation Engine' panel.")
    print("3. Click 'Generate Data' then 'Download CSV'.")
    print("4. Move the file to this folder: ", os.getcwd())
    exit()

data = pd.read_csv('fso_dataset.csv')

# Define features (X) and target (y)
FEATURES = ['distance_km', 'wavelength_nm', 'tx_power_mw', 'beam_div_mrad', 'attenuation_db_km']
TARGET = 'link_margin_db'

X = data[FEATURES]
y = data[TARGET]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples.")


# --- 2. Model A: Standard BP Neural Network ---
print("\n--- Training Model A: Standard BP (Adam optimizer) ---")

def create_bp_model(input_shape):
    """Creates a standard sequential Keras model."""
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer (regression)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create and train the model
bp_model = create_bp_model(X_train.shape[1])

start_time = time.time()
history_bp = bp_model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=0  # Set to 1 to see training progress
)
bp_train_time = time.time() - start_time
print(f"Standard BP trained in {bp_train_time:.2f} seconds.")


# --- 3. Model B: Hybrid Optimizer (Custom PSO-BP) ---
print("\n--- Training Model B: Hybrid BP (PSO optimizer) ---")
print("Note: Calculating fitness for particle swarm...")

# Get model structure details
keras_model = create_bp_model(X_train.shape[1])
shapes = [w.shape for w in keras_model.get_weights()]
dims = sum(np.prod(s) for s in shapes)

def set_model_weights(model, flat_weights):
    """Helper function to distribute flat weights back into the model layers."""
    idx = 0
    new_weights = []
    for shape in shapes:
        size = np.prod(shape)
        # Explicitly cast to float32 for TensorFlow compatibility
        w = flat_weights[idx:idx + size].reshape(shape).astype(np.float32)
        new_weights.append(w)
        idx += size
    model.set_weights(new_weights)

def pso_fitness_func(positions):
    """
    Evaluates MSE for a batch of particles (weight sets).
    """
    n_particles = positions.shape[0]
    mse_values = []

    # Note: In a production environment, we would parallelize this.
    # For this demo, we iterate to keep logic clear.
    for i in range(n_particles):
        set_model_weights(keras_model, positions[i])
        # Predict on a subset (batch) of data to speed up PSO
        # Using full dataset is better for accuracy but slower
        y_pred = keras_model.predict(X_train, batch_size=1024, verbose=0)
        mse = mean_squared_error(y_train, y_pred)
        mse_values.append(mse)
        
    return np.array(mse_values)

# Initialize our Custom PSO
# Fewer iterations/particles for speed in this demo
optimizer = CustomPSO(
    n_particles=15, 
    dimensions=dims, 
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
    bounds=(-1.0, 1.0)
)

# Run Optimization
start_time = time.time()
best_cost, best_weights = optimizer.optimize(pso_fitness_func, iters=15)
pso_opt_time = time.time() - start_time
print(f"PSO finished in {pso_opt_time:.2f}s. Best MSE found: {best_cost:.4f}")

# Apply best weights to a fresh model
hybrid_model = create_bp_model(X_train.shape[1])
set_model_weights(hybrid_model, best_weights)

# Fine-tune with Adam
print("Fine-tuning PSO weights with Adam optimizer...")
start_time = time.time()
history_hybrid = hybrid_model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=0
)
hybrid_train_time = time.time() - start_time
print(f"Hybrid BP fine-tuned in {hybrid_train_time:.2f} seconds.")


# --- 4. Evaluate and Compare Models ---
print("\n--- Model Evaluation ---")

# A. Standard BP Model
y_pred_bp = bp_model.predict(X_test).flatten()
mse_bp = mean_squared_error(y_test, y_pred_bp)
print(f"Model A (Standard BP) Test MSE: {mse_bp:.4f}")

# B. Hybrid PSO-BP Model
y_pred_hybrid = hybrid_model.predict(X_test).flatten()
mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
print(f"Model B (Hybrid PSO-BP) Test MSE: {mse_hybrid:.4f}")

improvement = ((mse_bp - mse_hybrid) / mse_bp) * 100
print(f"\nHybrid model shows a {improvement:.2f}% improvement in MSE.")

# --- 5. Generate Results Plot ---
print("Generating results plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Helper for limits
d_min = min(y.min(), y_pred_bp.min(), y_pred_hybrid.min())
d_max = max(y.max(), y_pred_bp.max(), y_pred_hybrid.max())

# Plot 1
ax1.scatter(y_test, y_pred_bp, alpha=0.3, label=f"MSE: {mse_bp:.4f}")
ax1.plot([d_min, d_max], [d_min, d_max], 'r--', lw=2, label="Perfect Prediction")
ax1.set_xlabel("True Link Margin (dB)")
ax1.set_ylabel("Predicted Link Margin (dB)")
ax1.set_title("Model A: Standard BP")
ax1.legend()
ax1.grid(True)

# Plot 2
ax2.scatter(y_test, y_pred_hybrid, alpha=0.3, color='orange', label=f"MSE: {mse_hybrid:.4f}")
ax2.plot([d_min, d_max], [d_min, d_max], 'r--', lw=2, label="Perfect Prediction")
ax2.set_xlabel("True Link Margin (dB)")
ax2.set_ylabel("Predicted Link Margin (dB)")
ax2.set_title("Model B: Hybrid PSO-BP")
ax2.legend()
ax2.grid(True)

fig.suptitle(f"FSO Model Comparison\nHybrid Improvement: {improvement:.2f}%", fontsize=16)
plt.tight_layout()
plt.savefig('results_plot.png')
print(f"\nDone! Results plot saved as 'results_plot.png'.")