# Interactive Digit Recognition GUI

A Tkinter graphical application for drawing digits freehand and running real-time inference with trained Multilayer Perceptron models (such as `models/ej3_mlp_optimized.pkl`).

## Features

1. **Graphical Model Picker**: Select any `.pkl` model file using the built-in Tkinter file chooser (no CLI arguments needed). Automatically loads model metadata, input standardization parameters ($\mu, \sigma$), and network layer architecture.
2. **Freehand Drawing Canvas**: Draw digits with your mouse on an intuitive 280×280 canvas with adjustable stroke thickness, eraser tool, and canvas clearing.
3. **Accuracy-Maximizing 28×28 Downscaling**:
   - **Smart Centering & CoM Alignment (MNIST Standard)**: Crops bounding box, resizes to a 20×20 box preserving aspect ratio, and centers the digit by Center of Mass (CoM) inside a 28×28 matrix with Lanczos anti-aliasing.
   - **Direct 28×28 Downsampling**: Toggleable mode for direct resizing.
   - **Real-Time 28×28 Preview**: Shows the exact matrix passed to the perceptron.
4. **10-Class Probability Output & Highlight**: Computes probabilities across all 10 digits (0 to 9), displays graphical probability bars, and highlights the top prediction.

## How to Run

From the project root:

```bash
uv run python interactive-digits/main.py
```

Or from inside the directory:

```bash
cd interactive-digits
uv run python main.py
```
