# train_esp.py — Runs on your LAPTOP
#
# Purpose: Trains a CNN to classify rock/paper/scissors from 32x32 grayscale images.
#          Uses training data collected by collect_esp.py (NumPy .npy files).
#          Exports model weights (model_esp32.bin) and inference code (cnn_model.py)
#          for deployment on the ESP32 microcontroller.
#
# CNN Architecture: 3x Conv2D(12 filters) + MaxPool + Dense(24) + Dense(3)
# Total parameters: 7,443 (~29.8 KB) — small enough for ESP32
#
# Usage: python3 train_esp.py
#
# Author: Shravani Poman (with assistance from Claude AI)
# Course: AI for Engineers, Spring 2026

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import struct

# --- Training Configuration ---
DATA_DIR = "training_data_esp"            # Folder with .npy training images
CLASSES = ["rock", "paper", "scissors"]   # Class labels (index 0, 1, 2)
IMG_SIZE = 32                             # Input image size (32x32 pixels)
EPOCHS = 50                               # Number of training passes
BATCH_SIZE = 16                           # Images per gradient update

def load_dataset():
    """Load all .npy training images and labels from disk.
    Each .npy file is a 32x32 float32 array (0.0-1.0) created by collect_esp.py.
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
            pixels = np.load(os.path.join(folder, fname))
            if pixels.shape == (IMG_SIZE, IMG_SIZE):
                images.append(pixels)
                labels.append(class_idx)  # 0=rock, 1=paper, 2=scissors
    # Reshape: (N, 32, 32) -> (N, 32, 32, 1) for CNN channel dimension
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    print(f"\nTotal: {len(images)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {np.sum(labels == i)}")
    return images, labels

def augment(images, labels):
    """Apply data augmentation to expand training set ~5x.
    Helps the model generalize to variations in hand position and lighting.
    
    Four techniques applied to each image:
    1. Horizontal flip — simulates left/right hand
    2. Gaussian noise (std=0.03) — handles camera sensor noise
    3. Brightness shift (±0.08) — handles lighting variation
    4. Spatial shift (±1 pixel) — handles positioning variation"""
    aug_imgs = list(images)
    aug_lbls = list(labels)
    for img, lbl in zip(images, labels):
        # 1. Horizontal flip
        aug_imgs.append(np.fliplr(img))
        aug_lbls.append(lbl)
        # 2. Add Gaussian noise
        noisy = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)
        aug_imgs.append(np.clip(noisy, 0, 1))
        aug_lbls.append(lbl)
        # 3. Random brightness shift
        bright = img + np.random.uniform(-0.08, 0.08)
        aug_imgs.append(np.clip(bright, 0, 1).astype(np.float32))
        aug_lbls.append(lbl)
        # 4. Small spatial shift (±1 pixel)
        shifted = np.roll(img, np.random.randint(-1, 2), axis=0)
        shifted = np.roll(shifted, np.random.randint(-1, 2), axis=1)
        aug_imgs.append(shifted)
        aug_lbls.append(lbl)
    return np.array(aug_imgs), np.array(aug_lbls)

def build_model():
    """Build compact CNN architecture for ESP32 deployment.
    
    Architecture:
    - 3 Conv2D blocks: 12 filters each (small to fit ESP32 memory)
      Each block: Conv2D(3x3, ReLU, same padding) + MaxPool(2x2)
      Progressively extracts features: edges -> shapes -> hand patterns
    - Flatten: 4x4x12 = 192 values
    - Dense(24, ReLU) with 30% dropout: learns decision boundaries
    - Dense(3, Softmax): outputs probabilities for rock/paper/scissors
    
    Total: 7,443 parameters (~29.8 KB)"""
    model = keras.Sequential([
        # Block 1: Detect edges and textures (32x32 -> 16x16)
        layers.Conv2D(12, (3, 3), activation='relu', padding='same',
                      input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        # Block 2: Combine into shapes (16x16 -> 8x8)
        layers.Conv2D(12, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Block 3: Capture hand contour patterns (8x8 -> 4x4)
        layers.Conv2D(12, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Classification head
        layers.Flatten(),           # 4x4x12 = 192 values
        layers.Dropout(0.3),        # Prevent overfitting during training
        layers.Dense(24, activation='relu'),  # Decision boundary learning
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 class probabilities
    ])
    # Adam optimizer with sparse cross-entropy loss (labels are integers, not one-hot)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def export_for_esp32(model):
    """Export trained model for ESP32 deployment.
    
    Creates two files:
    1. model_esp32.bin — All weights as sequential float32 values
       Order: conv0 kernel+bias, conv1 kernel+bias, conv2 kernel+bias,
              dense0 weights+bias, dense1 weights+bias
    
    2. cnn_model.py — MicroPython inference engine that reimplements
       the CNN forward pass using only struct and math modules.
       Auto-generated with correct architecture parameters."""
    print("\n--- Exporting for ESP32 ---")
    conv_layers = [l for l in model.layers if 'conv2d' in l.name]
    dense_layers = [l for l in model.layers if 'dense' in l.name]

    # Write all weights to binary file as raw float32
    with open("model_esp32.bin", "wb") as f:
        for i, layer in enumerate(conv_layers):
            k, b = layer.get_weights()  # k=kernels, b=biases
            print(f"  Conv{i}: {k.shape}, {b.shape} = {(k.size+b.size)*4} bytes")
            f.write(k.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())
        for i, layer in enumerate(dense_layers):
            w, b = layer.get_weights()  # w=weights, b=biases
            print(f"  Dense{i}: {w.shape}, {b.shape} = {(w.size+b.size)*4} bytes")
            f.write(w.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())

    # Extract architecture parameters for code generation
    n_filters = [l.get_weights()[0].shape[3] for l in conv_layers]
    dense_shapes = [(l.get_weights()[0].shape, l.get_weights()[1].shape) for l in dense_layers]
    flatten_size = 4 * 4 * n_filters[-1]  # 4x4 spatial after 3 pooling layers

    # Generate MicroPython inference script
    # This reimplements the CNN forward pass without TensorFlow
    script = f'''# cnn_model.py — Auto-generated by train_esp.py
# CNN inference engine for ESP32 on-device classification
# Architecture: Conv2D(12)->Pool->Conv2D(12)->Pool->Conv2D(12)->Pool->Dense(24)->Dense(3)
# Reads weights from model_esp32.bin (~29.8 KB)
#
# Author: Shravani Poman (with assistance from Claude AI)

import struct
import math

CLASSES = ["rock", "paper", "scissors"]

def relu(x):
    """ReLU activation: max(0, x). Introduces non-linearity."""
    return max(0.0, x)

def softmax(values):
    """Convert raw scores to probability distribution (sums to 1.0).
    Subtracts max for numerical stability."""
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
        """Load all model weights from binary file.
        Weights stored as sequential float32 values."""
        with open("model_esp32.bin", "rb") as f:
            data = f.read()
        offset = 0
        def read_floats(n):
            nonlocal offset
            vals = struct.unpack_from("<" + "f" * n, data, offset)
            offset += n * 4
            return list(vals)
        # Convolutional layer weights
        self.conv0_k = read_floats(3*3*1*{n_filters[0]})
        self.conv0_b = read_floats({n_filters[0]})
        self.conv1_k = read_floats(3*3*{n_filters[0]}*{n_filters[1]})
        self.conv1_b = read_floats({n_filters[1]})
        self.conv2_k = read_floats(3*3*{n_filters[1]}*{n_filters[2]})
        self.conv2_b = read_floats({n_filters[2]})
        # Dense layer weights
        self.dense0_w = read_floats({dense_shapes[0][0][0]}*{dense_shapes[0][0][1]})
        self.dense0_b = read_floats({dense_shapes[0][1][0]})
        self.dense1_w = read_floats({dense_shapes[1][0][0]}*{dense_shapes[1][0][1]})
        self.dense1_b = read_floats({dense_shapes[1][1][0]})

    def conv2d(self, inp, in_h, in_w, in_ch, kernel, bias, out_ch):
        """3x3 convolution with same padding and ReLU activation.
        TensorFlow kernel layout: [ky][kx][in_ch][out_ch]"""
        out = [0.0] * (out_ch * in_h * in_w)
        for oc in range(out_ch):
            for y in range(in_h):
                for x in range(in_w):
                    val = bias[oc]
                    for ky in range(3):
                        for kx in range(3):
                            iy = y + ky - 1
                            ix = x + kx - 1
                            if 0 <= iy < in_h and 0 <= ix < in_w:
                                for ic in range(in_ch):
                                    k_idx = ((ky * 3 + kx) * in_ch + ic) * out_ch + oc
                                    p_idx = (iy * in_w + ix) * in_ch + ic
                                    val += inp[p_idx] * kernel[k_idx]
                    idx = (y * in_w + x) * out_ch + oc
                    out[idx] = relu(val)
        return out

    def maxpool2d(self, inp, in_h, in_w, channels):
        """2x2 max pooling — reduces spatial dimensions by half."""
        out_h = in_h // 2
        out_w = in_w // 2
        out = [0.0] * (out_h * out_w * channels)
        for y in range(out_h):
            for x in range(out_w):
                for c in range(channels):
                    max_val = -999999.0
                    for dy in range(2):
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
        """Fully connected layer: output = sum(input * weights) + bias."""
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
        """Run CNN forward pass on 32x32 grayscale image.
        pixels: flat list of 1024 floats (0.0-1.0)
        Returns: (class_name, confidence, [rock, paper, scissors probs])"""
        x = list(pixels)
        # Conv blocks: extract features at decreasing spatial resolution
        x = self.conv2d(x, 32, 32, 1, self.conv0_k, self.conv0_b, {n_filters[0]})
        x = self.maxpool2d(x, 32, 32, {n_filters[0]})
        x = self.conv2d(x, 16, 16, {n_filters[0]}, self.conv1_k, self.conv1_b, {n_filters[1]})
        x = self.maxpool2d(x, 16, 16, {n_filters[1]})
        x = self.conv2d(x, 8, 8, {n_filters[1]}, self.conv2_k, self.conv2_b, {n_filters[2]})
        x = self.maxpool2d(x, 8, 8, {n_filters[2]})
        # Classification
        x = self.dense(x, self.dense0_w, self.dense0_b, {flatten_size}, {dense_shapes[0][0][1]}, use_relu=True)
        x = self.dense(x, self.dense1_w, self.dense1_b, {dense_shapes[1][0][0]}, 3, use_relu=False)
        probs = softmax(x)
        best_idx = 0
        best_val = probs[0]
        for i in range(1, 3):
            if probs[i] > best_val:
                best_val = probs[i]
                best_idx = i
        return CLASSES[best_idx], best_val, probs
'''

    with open("cnn_model.py", "w") as f:
        f.write(script)
    print("  Saved cnn_model.py")
    model.save("rps_tiny_model.keras")
    print("  Saved rps_tiny_model.keras")

def main():
    """Main training pipeline: load -> augment -> train -> evaluate -> export."""
    print("=" * 50)
    print("  Training CNN for ESP32 on-device classification")
    print("=" * 50)

    # Step 1: Load training data
    images, labels = load_dataset()
    if len(images) == 0:
        print("No data! Run collect_esp.py first.")
        return

    # Step 2: Augment dataset (~5x expansion)
    print("\n--- Augmenting ---")
    images, labels = augment(images, labels)
    print(f"After augmentation: {len(images)}")

    # Step 3: Shuffle randomly
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # Step 4: Split into train (80%) and test (20%)
    split = int(len(images) * 0.8)
    train_imgs, test_imgs = images[:split], images[split:]
    train_lbls, test_lbls = labels[:split], labels[split:]

    # Step 5: Build and train model
    model = build_model()
    model.summary()

    model.fit(train_imgs, train_lbls, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(test_imgs, test_lbls), verbose=1)

    # Step 6: Evaluate
    loss, acc = model.evaluate(test_imgs, test_lbls, verbose=0)
    print(f"\nTest Accuracy: {acc * 100:.1f}%")

    # Per-class accuracy
    preds = np.argmax(model.predict(test_imgs, verbose=0), axis=1)
    for i, cls in enumerate(CLASSES):
        mask = test_lbls == i
        if np.sum(mask) > 0:
            a = np.sum(preds[mask] == i) / np.sum(mask)
            print(f"  {cls}: {a * 100:.1f}%")

    # Step 7: Export for ESP32
    export_for_esp32(model)
    print(f"\nDONE! Accuracy: {acc*100:.1f}%")
    print("\nUpload these to ESP32:")
    print("  - model_esp32.bin")
    print("  - cnn_model.py")

if __name__ == "__main__":
    main()
