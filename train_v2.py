# train_v2.py — Runs on your LAPTOP
# Purpose: Trains a Convolutional Neural Network (CNN) to classify hand gestures
#          as rock, paper, or scissors from 32x32 grayscale images.
#
# Input: NumPy .npy files from training_data_v2/ (created by collect_v2.py)
# Output: rps_tiny_model.keras (trained Keras model for laptop classification)
#         model_esp32.bin (binary weights for ESP32 on-device classification)
#         cnn_model.py (MicroPython inference engine for ESP32)
#
# CNN Architecture: 3x Conv2D(12 filters) + MaxPool → Dense(24) → Dense(3)
# Total parameters: ~7,443 (~30KB) — small enough for ESP32
#
# Author: Shravani Poman (with assistance from Claude AI)


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import struct

# --- Training Configuration ---
DATA_DIR = "training_data_v2"            # Folder with .npy training images
CLASSES = ["rock", "paper", "scissors"]  # Class labels (index 0, 1, 2)
IMG_SIZE = 32                            # Image dimensions (32x32 pixels)
EPOCHS = 50                              # Number of training passes through the data
BATCH_SIZE = 16                          # Images processed per gradient update

def load_dataset():
    """Load all .npy training images and their labels from disk.
    Each .npy file is a 32x32 float32 array (pixel values 0.0-1.0)
    created by collect_v2.py with matching preprocessing pipeline.
    Returns: (images array [N,32,32,1], labels array [N])"""
    images = []
    labels = []
    for class_idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        print(f"Loading {cls}: {len(files)} images")
        for fname in files:
            filepath = os.path.join(folder, fname)
            pixels = np.load(filepath)  # Load 32x32 float array
            if pixels.shape == (IMG_SIZE, IMG_SIZE):
                images.append(pixels)
                labels.append(class_idx)  # 0=rock, 1=paper, 2=scissors
    # Reshape to add channel dimension: (N, 32, 32) → (N, 32, 32, 1)
    # The "1" channel indicates grayscale (RGB would be 3 channels)
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    print(f"\nTotal: {len(images)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {np.sum(labels == i)}")
    return images, labels

def augment(images, labels):
    """Apply data augmentation to increase training set size ~5x.
    Augmentation creates modified copies of each image to help the model
    generalize to slight variations in hand position, lighting, and noise.
    
    Four augmentation techniques applied to each image:
    1. Horizontal flip — simulates left/right hand
    2. Gaussian noise — improves robustness to camera sensor noise
    3. Brightness shift — handles slight lighting changes
    4. Spatial shift — handles small positioning differences
    """
    aug_imgs = list(images)
    aug_lbls = list(labels)
    for img, lbl in zip(images, labels):
        # 1. Horizontal flip (mirror image left-right)
        aug_imgs.append(np.fliplr(img))
        aug_lbls.append(lbl)
        
        # 2. Add random Gaussian noise (std dev = 0.03)
        noisy = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)
        aug_imgs.append(np.clip(noisy, 0, 1))  # Clip to valid range
        aug_lbls.append(lbl)
        
        # 3. Random brightness shift (±8% of full range)
        bright = img + np.random.uniform(-0.08, 0.08)
        aug_imgs.append(np.clip(bright, 0, 1).astype(np.float32))
        aug_lbls.append(lbl)
        
        # 4. Small spatial shift (±1 pixel in each direction)
        # np.roll wraps pixels around — acceptable for small shifts
        shifted = np.roll(img, np.random.randint(-1, 2), axis=0)  # Vertical shift
        shifted = np.roll(shifted, np.random.randint(-1, 2), axis=1)  # Horizontal shift
        aug_imgs.append(shifted)
        aug_lbls.append(lbl)
    return np.array(aug_imgs), np.array(aug_lbls)

def build_model():
    """Build the CNN architecture for rock/paper/scissors classification.
    
    Architecture (designed to be small enough for ESP32):
    - 3 Convolutional blocks: Conv2D(12 filters, 3x3) + MaxPool(2x2)
      Progressively extracts features: edges → shapes → hand patterns
      Using 12 filters (not typical 32/64) to keep model under 30KB
    - Flatten: Converts 4x4x12 feature maps to 192-element vector
    - Dense(24, ReLU): Learns decision boundaries from features
    - Dropout(0.3): Randomly disables 30% of neurons during training
      to prevent overfitting (forces redundant learning)
    - Dense(3, Softmax): Output layer — probabilities for each class
      Softmax ensures outputs sum to 1.0 (probability distribution)
    
    Input:  32x32x1 grayscale image (normalized 0.0-1.0)
    Output: 3 probabilities [rock, paper, scissors]
    """
    model = keras.Sequential([
        # Block 1: Detect basic edges and textures
        # 12 filters scan 3x3 windows across the 32x32 input
        # 'same' padding preserves spatial dimensions
        # ReLU activation: max(0, x) — introduces non-linearity
        layers.Conv2D(12, (3, 3), activation='relu', padding='same',
                      input_shape=(IMG_SIZE, IMG_SIZE, 1)),  # Output: 32x32x12
        layers.MaxPooling2D((2, 2)),  # Take max of each 2x2 block → 16x16x12

        # Block 2: Combine edges into shapes (curves, lines)
        layers.Conv2D(12, (3, 3), activation='relu', padding='same'),  # Output: 16x16x12
        layers.MaxPooling2D((2, 2)),  # → 8x8x12

        # Block 3: Capture high-level hand contour patterns
        layers.Conv2D(12, (3, 3), activation='relu', padding='same'),  # Output: 8x8x12
        layers.MaxPooling2D((2, 2)),  # → 4x4x12

        # Classification head
        layers.Flatten(),       # 4x4x12 = 192 values in a single vector
        layers.Dropout(0.3),    # Drop 30% of connections to prevent overfitting
        layers.Dense(24, activation='relu'),  # 24 neurons learn decision boundaries
        layers.Dropout(0.3),    # Another dropout layer for regularization
        layers.Dense(3, activation='softmax')  # 3 output probabilities
    ])
    
    # Compile with Adam optimizer (adaptive learning rate) and cross-entropy loss
    # Sparse categorical = labels are integers (0,1,2) not one-hot encoded
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def export_for_esp32(model):
    """Export the trained model for deployment on the ESP32 microcontroller.
    
    Creates two files:
    1. model_esp32.bin — Binary file containing all weights as float32 values
       Weights are written in order: conv0 kernel+bias, conv1 kernel+bias,
       conv2 kernel+bias, dense0 weights+bias, dense1 weights+bias
    
    2. cnn_model.py — MicroPython script that reimplements the CNN forward pass
       using pure Python (no TensorFlow). Reads weights from model_esp32.bin
       and implements conv2d, maxpool, dense, relu, and softmax operations.
       This allows classification directly on the ESP32.
    """
    print("\n--- Exporting for ESP32 ---")
    conv_layers = [l for l in model.layers if 'conv2d' in l.name]
    dense_layers = [l for l in model.layers if 'dense' in l.name]

    # Write all weights to a single binary file as raw float32 values
    with open("model_esp32.bin", "wb") as f:
        for i, layer in enumerate(conv_layers):
            k, b = layer.get_weights()  # k=kernel weights, b=bias values
            print(f"  Conv{i}: {k.shape}, {b.shape} = {(k.size+b.size)*4} bytes")
            f.write(k.astype(np.float32).tobytes())  # Write kernel
            f.write(b.astype(np.float32).tobytes())  # Write bias
        for i, layer in enumerate(dense_layers):
            w, b = layer.get_weights()  # w=weight matrix, b=bias vector
            print(f"  Dense{i}: {w.shape}, {b.shape} = {(w.size+b.size)*4} bytes")
            f.write(w.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())

    # Extract architecture parameters for code generation
    n_filters = [l.get_weights()[0].shape[3] for l in conv_layers]  # Filters per conv layer
    dense_shapes = [(l.get_weights()[0].shape, l.get_weights()[1].shape) for l in dense_layers]
    flatten_size = 4 * 4 * n_filters[-1]  # 4x4 spatial size after 3 pooling layers × filters

    # Generate MicroPython inference script using f-string template
    # The generated code implements the same CNN forward pass without TensorFlow
    script = f'''# cnn_model.py — Auto-generated MicroPython CNN inference engine
# Implements the trained CNN forward pass for on-device classification
# Reads weights from model_esp32.bin
#
# Architecture: Conv2D(12)→Pool→Conv2D(12)→Pool→Conv2D(12)→Pool→Dense(24)→Dense(3)
# Author: Generated by train_v2.py

import struct
import math

CLASSES = ["rock", "paper", "scissors"]

def relu(x):
    """ReLU activation: returns max(0, x). Introduces non-linearity."""
    return max(0.0, x)

def softmax(values):
    """Softmax: converts raw scores to probability distribution (sums to 1.0).
    Subtracts max value first for numerical stability (prevents overflow)."""
    max_val = max(values)
    exp_vals = [math.exp(v - max_val) for v in values]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]

class TinyCNN:
    def __init__(self):
        print("Loading CNN model...")
        self._load_weights()
        print("Model loaded!")
    
    def _load_weights(self):
        """Read all model weights from binary file into memory.
        Weights are stored as sequential float32 values."""
        with open("model_esp32.bin", "rb") as f:
            data = f.read()
        offset = 0
        def read_floats(n):
            nonlocal offset
            vals = struct.unpack_from("<" + "f" * n, data, offset)
            offset += n * 4
            return list(vals)
        # Load convolutional layer weights (kernel shape: H x W x in_ch x out_ch)
        self.conv0_k = read_floats(3*3*1*{n_filters[0]})
        self.conv0_b = read_floats({n_filters[0]})
        self.conv1_k = read_floats(3*3*{n_filters[0]}*{n_filters[1]})
        self.conv1_b = read_floats({n_filters[1]})
        self.conv2_k = read_floats(3*3*{n_filters[1]}*{n_filters[2]})
        self.conv2_b = read_floats({n_filters[2]})
        # Load dense layer weights (shape: in_size x out_size)
        self.dense0_w = read_floats({dense_shapes[0][0][0]}*{dense_shapes[0][0][1]})
        self.dense0_b = read_floats({dense_shapes[0][1][0]})
        self.dense1_w = read_floats({dense_shapes[1][0][0]}*{dense_shapes[1][0][1]})
        self.dense1_b = read_floats({dense_shapes[1][1][0]})

    def conv2d(self, inp, in_h, in_w, in_ch, kernel, bias, out_ch):
        """2D convolution with 3x3 kernel, same padding, ReLU activation.
        For each output position, computes weighted sum of 3x3 neighborhood
        across all input channels, adds bias, applies ReLU.
        TensorFlow kernel layout: [ky][kx][in_channel][out_channel]"""
        out = [0.0] * (out_ch * in_h * in_w)
        for oc in range(out_ch):           # For each output filter
            for y in range(in_h):          # For each output row
                for x in range(in_w):      # For each output column
                    val = bias[oc]         # Start with bias
                    for ky in range(3):    # 3x3 kernel height
                        for kx in range(3):  # 3x3 kernel width
                            iy = y + ky - 1  # Input y with padding offset
                            ix = x + kx - 1  # Input x with padding offset
                            if 0 <= iy < in_h and 0 <= ix < in_w:  # Bounds check (zero padding)
                                for ic in range(in_ch):  # Sum across input channels
                                    k_idx = ((ky * 3 + kx) * in_ch + ic) * out_ch + oc
                                    p_idx = (iy * in_w + ix) * in_ch + ic
                                    val += inp[p_idx] * kernel[k_idx]
                    idx = (y * in_w + x) * out_ch + oc
                    out[idx] = relu(val)   # Apply ReLU activation
        return out

    def maxpool2d(self, inp, in_h, in_w, channels):
        """2x2 max pooling: takes maximum value from each 2x2 block.
        Reduces spatial dimensions by half (e.g. 32x32 -> 16x16)."""
        out_h = in_h // 2
        out_w = in_w // 2
        out = [0.0] * (out_h * out_w * channels)
        for y in range(out_h):
            for x in range(out_w):
                for c in range(channels):
                    max_val = -999999.0
                    for dy in range(2):    # 2x2 pooling window
                        for dx in range(2):
                            iy = y * 2 + dy
                            ix = x * 2 + dx
                            idx = (iy * in_w + ix) * channels + c
                            if inp[idx] > max_val:
                                max_val = inp[idx]
                    out_idx = (y * out_w + x) * channels + c
                    out[out_idx] = max_val
        return out

    def dense(self, inp, weights, bias, in_size, out_size, use_relu=True):
        """Fully connected (dense) layer: each output = sum(inputs * weights) + bias.
        Optionally applies ReLU activation."""
        out = []
        for o in range(out_size):
            val = bias[o]
            for i in range(in_size):
                val += inp[i] * weights[i * out_size + o]
            if use_relu:
                val = relu(val)
            out.append(val)
        return out

    def predict(self, pixels):
        """Run full CNN forward pass on a 32x32 grayscale image.
        pixels: flat list of 1024 floats (row-major, 0.0-1.0)
        Returns: (class_name, confidence, [rock_prob, paper_prob, scissors_prob])"""
        x = list(pixels)
        # Conv block 1: (32,32,1) -> (32,32,12) -> (16,16,12)
        x = self.conv2d(x, 32, 32, 1, self.conv0_k, self.conv0_b, {n_filters[0]})
        x = self.maxpool2d(x, 32, 32, {n_filters[0]})
        # Conv block 2: (16,16,12) -> (16,16,12) -> (8,8,12)
        x = self.conv2d(x, 16, 16, {n_filters[0]}, self.conv1_k, self.conv1_b, {n_filters[1]})
        x = self.maxpool2d(x, 16, 16, {n_filters[1]})
        # Conv block 3: (8,8,12) -> (8,8,12) -> (4,4,12)
        x = self.conv2d(x, 8, 8, {n_filters[1]}, self.conv2_k, self.conv2_b, {n_filters[2]})
        x = self.maxpool2d(x, 8, 8, {n_filters[2]})
        # Flatten (4*4*12=192) -> Dense(24) -> Dense(3)
        x = self.dense(x, self.dense0_w, self.dense0_b, {flatten_size}, {dense_shapes[0][0][1]}, use_relu=True)
        x = self.dense(x, self.dense1_w, self.dense1_b, {dense_shapes[1][0][0]}, 3, use_relu=False)
        # Convert raw scores to probabilities
        probs = softmax(x)
        # Find class with highest probability
        best_idx = 0
        best_val = probs[0]
        for i in range(1, 3):
            if probs[i] > best_val:
                best_val = probs[i]
                best_idx = i
        return CLASSES[best_idx], best_val, probs
'''

    # Write the generated inference script
    with open("cnn_model.py", "w") as f:
        f.write(script)
    print("  Saved cnn_model.py")
    
    # Save the Keras model for laptop-side classification
    model.save("rps_tiny_model.keras")
    print("  Saved rps_tiny_model.keras")

def main():
    """Main training pipeline: load data → augment → train → evaluate → export."""
    print("=" * 50)
    print("  Training CNN v2 (matched pipeline)")
    print("=" * 50)

    # Step 1: Load training data from .npy files
    images, labels = load_dataset()
    if len(images) == 0:
        print("No data! Run collect_v2.py first.")
        return

    # Step 2: Augment dataset (~5x expansion)
    print("\n--- Augmenting ---")
    images, labels = augment(images, labels)
    print(f"After augmentation: {len(images)}")

    # Step 3: Shuffle data randomly for training
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # Step 4: Split into training (80%) and test (20%) sets
    split = int(len(images) * 0.8)
    train_imgs, test_imgs = images[:split], images[split:]
    train_lbls, test_lbls = labels[:split], labels[split:]

    # Step 5: Build and display model architecture
    model = build_model()
    model.summary()  # Print layer-by-layer parameter counts

    # Step 6: Train the model
    model.fit(train_imgs, train_lbls, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(test_imgs, test_lbls), verbose=1)

    # Step 7: Evaluate on test set
    loss, acc = model.evaluate(test_imgs, test_lbls, verbose=0)
    print(f"\nTest Accuracy: {acc * 100:.1f}%")

    # Per-class accuracy breakdown
    preds = np.argmax(model.predict(test_imgs, verbose=0), axis=1)
    for i, cls in enumerate(CLASSES):
        mask = test_lbls == i
        if np.sum(mask) > 0:
            a = np.sum(preds[mask] == i) / np.sum(mask)
            print(f"  {cls}: {a * 100:.1f}%")

    # Step 8: Export model for ESP32 and save Keras model
    export_for_esp32(model)
    print(f"\nDONE! Accuracy: {acc*100:.1f}%")

if __name__ == "__main__":
    main()
