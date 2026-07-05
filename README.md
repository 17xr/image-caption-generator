# 🖼️ Image Caption Generator

An elegant, lightweight (~168M parameters) image captioning model built completely from scratch in PyTorch for learning purposes. Inspired by the multimodal design principles of [**Florence-2**](https://arxiv.org/pdf/2311.06242) and the streamlined decoder-only architecture of [**Qwen 3**](https://arxiv.org/pdf/2505.09388), this project provides an end-to-end framework for styled image captioning. 

The primary goal of this repository is educational—demonstrating how to parse vision features, construct token-level causal attention masks for cross-modality, implement high-performance inference with KV caching, and orchestrate optimized training loops using pure PyTorch.

## ✨ Key Features

### 1. Hybrid Multimodal Architecture
* **Vision Encoder:** Uses a **DINOv3 ViT backbone** (`dinov3-vits16plus`) paired with a space-to-channel (`PixelUnshuffle`) projection bridge to map patch-level features into the text embedding space.
* **Decoder-Only Text Transformer:** A custom 12-layer, 12-head decoder inspired by Qwen 3, utilizing **RMSNorm**, **RoPE**, and **SwiGLU** activation blocks.
* **Unified Multimodal Attention Masking:** Imploys a specialized mask allowing prefix image tokens to attend to each other fully, while enforcing strict causal autoregressive attention for style and text tokens.

### 2. Controllable Caption Styling
Trained on custom MS COCO captions generated using a local VLM, the model parses special prefix conditioning tokens to support three distinct output styles:
* `[CONCISE]`: Short, direct, and punchy visual summaries.
* `[DESCRIPTIVE]`: Detailed descriptions capturing textures, fine-grained details, and lighting conditions.
* `[NARRATIVE]`: Immersive, context-rich descriptions leaning into a storytelling perspective.

### 3. Production-Ready Inference Pipeline
* **KV Caching Support:** Tracks past key and value matrices to shift generation time complexity from $O(N^2)$ to $O(1)$ per token for rapid text decoding.
* **Advanced Sampling Engine:** Configurable parameters for temperature scaling, `top_k` filtering, nucleus sampling, dynamic repetition penalties, and custom n-gram blocking.

### 4. Advanced Training Implementation
Built entirely from scratch in pure PyTorch with advanced learning routines:
* **Gradient Accumulation:** Simulates large effective batch sizes seamlessly on limited VRAM.
* **Warmup & Cosine Schedulers:** Uses a `SequentialLR` layout for linear learning rate warmup followed by cosine annealing.
* **Decoupled Weight Decay:** Selectively applies L2 regularization to weights while excluding biases and normalization parameters.
* **Label Smoothing:** Built directly into the cross-entropy loss objective to prevent overfitting.

## 📂 File Structure

```text
├── backend
│   ├── app
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── config.py
│   │   └── main.py             # FastAPI inference service
│   ├── architecture
│   │   ├── __init__.py
│   │   └── transformer.py      # Model layers
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── data
│   ├── concise
│   │   ├── coco_train.csv
│   │   ├── coco_valid.csv
│   │   └── coco_test.csv
│   ├── descriptive
│   │   ├── coco_train.csv
│   │   ├── coco_valid.csv
│   │   └── coco_test.csv
│   └── narrative
│       ├── coco_train.csv
│       ├── coco_valid.csv
│       └── coco_test.csv
├── frontend
│   └── src
│       └── main.py             # Streamlit web interface
├── models
│   └── weights.pt              # Model weights
├── notebooks
│   └── training.ipynb          # End-to-end training notebook
├── LICENSE
├── README.md
└── requirements.txt            # Python environment dependencies
```

## 🚀 Installation & Setup

This repository uses `uv`, an ultra-fast Python package installer and resolver, to ensure rapid, reproducible environment isolation.

### 1. Prerequisites
Make sure you have `uv` installed. If you don't have it yet, download it via curl or pip:
```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or alternatively via pip
pip install uv
```

### 2. Environment Isolation & Dependencies
Navigate to the root of your cloned repository to create a virtual environment using `uv`, activate it, and install the required packages:

```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (cmd):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install project dependencies from the requirements.txt file
uv pip install -r requirements.txt
```

---

## 💻 Running the Application

To interact with the image caption generator project, you must launch the FastAPI inference service and the Streamlit web interface separately from the root directory.

### 1. Start the Inference Service (FastAPI via Uvicorn)
The backend loads the PyTorch architecture configuration, references your local weights (`models/weights.pt`), and hosts a web endpoint using `uvicorn`:

```
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend Client (Streamlit)
Open a new terminal window or tab, ensure your virtual environment is active, and launch the user interface layer:

```
streamlit run frontend/src/main.py
```
This will automatically open the web interface in your default browser at `http://localhost:8501`. From here, you can upload images and check the generated multi-style captions!


## 🛠️ Tech Stack

* **Deep Learning:** PyTorch
* **Vision Backbone:** DINOv3 ViT (`dinov3-vits16plus`) via HuggingFace
* **Inference Service:** FastAPI + Uvicorn
* **Client Interface:** Streamlit
* **Environment & Package Management:** `uv` (Astral)
