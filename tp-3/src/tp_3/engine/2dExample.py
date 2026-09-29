import io
from contextlib import redirect_stdout
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from activation_functions.identity import Identity
from models.mlp import MLP  # Imports the MLP class provided in your codebase

def target_function(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """2D Damped Wave / Sinc Surface."""
    r = np.sqrt(x1**2 + x2**2)
    return np.sin(r) * np.exp(-0.1 * r)

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Generate Training Data (2D grid sampling)
    num_samples = 2500
    x1_vals = np.random.uniform(-3 * np.pi, 3 * np.pi, num_samples)
    x2_vals = np.random.uniform(-3 * np.pi, 3 * np.pi, num_samples)

    X = np.column_stack([x1_vals, x2_vals])
    y = target_function(x1_vals, x2_vals).reshape(-1, 1)

    # 2. Build Evaluation Grid (50x50 for smooth Plotly rendering)
    grid_size = 50
    g1 = np.linspace(-3 * np.pi, 3 * np.pi, grid_size)
    g2 = np.linspace(-3 * np.pi, 3 * np.pi, grid_size)
    G1, G2 = np.meshgrid(g1, g2)
    X_grid = np.column_stack([G1.ravel(), G2.ravel()])
    Y_true = target_function(G1, G2)

    # 3. Build the MLP Architecture
    network = MLP(
        layer_sizes=[2, 64, 64, 32, 1], 
        output_activation=Identity()
    )

    # 4. Train incrementally across non-linear milestones (~50 frames, dense at start)
    total_epochs = 2500
    num_frames = 50
    epoch_targets = np.unique(np.round(np.linspace(1, np.sqrt(total_epochs), num_frames) ** 2).astype(int))

    frames = []
    current_epoch = 0

    print("--- Training MLP on 2D Damped Wave Function ---")
    for target_epoch in epoch_targets:
        epochs_to_run = target_epoch - current_epoch
        
        with redirect_stdout(io.StringIO()):
            network.fit(X, y, epochs=epochs_to_run, lr=0.001, print_every=epochs_to_run)
        
        current_epoch = target_epoch
        loss = np.mean((network.forward(X) - y) ** 2)
        print(f"Epoch {current_epoch:4d}/{total_epochs} | Loss: {loss:.6f}")
        
        Y_pred = network.forward(X_grid).reshape(grid_size, grid_size)
        frames.append(go.Frame(
            data=[
                go.Surface(z=Y_true, x=G1, y=G2, colorscale='Viridis', showscale=False),
                go.Surface(z=Y_pred, x=G1, y=G2, colorscale='Plasma', showscale=False)
            ],
            name=f"Epoch {current_epoch}"
        ))

    # 5. Build Animated Plotly Figure
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=["Ground Truth: f(x, y)", "MLP Prediction"]
    )

    # Base initial state
    fig.add_trace(go.Surface(z=Y_true, x=G1, y=G2, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(z=frames[0].data[1].z, x=G1, y=G2, colorscale='Plasma', showscale=False), row=1, col=2)

    fig.frames = frames
    fig.update_layout(
        title="MLP 3D Surface Learning Progress Across Epochs",
        scene=dict(zaxis=dict(range=[-1.0, 1.0])),
        scene2=dict(zaxis=dict(range=[-1.0, 1.0])),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ]
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f.name,
                    "method": "animate"
                } for f in frames
            ]
        }]
    )

    # Save interactive animation to disk
    output_path = "mlp_learning_progress.html"
    fig.write_html(output_path)
    print(f"\nSaved interactive animation to: {output_path}")

    # 6. Print Sample Numerical Comparisons
    print("\n--- Sample Predictions ---")
    test_pts = np.array([[0.0, 0.0], [np.pi/2, 0.0], [np.pi, np.pi]])
    true_vals = target_function(test_pts[:, 0], test_pts[:, 1]).reshape(-1, 1)
    pred_vals = network.forward(test_pts)

    for i in range(len(test_pts)):
        print(f"Input: {test_pts[i]} | True: {true_vals[i][0]:.4f} | Pred: {pred_vals[i][0]:.4f}")
