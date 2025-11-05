#!/bin/bash

# Startup script for Swedish Audio Translator
# Sets environment variables to fix PyTorch hanging issues on macOS

# Activate virtual environment
source venv/bin/activate

# Set environment variables to prevent PyTorch from hanging on macOS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Disable PyTorch JIT compiler (can cause hangs)
export PYTORCH_JIT=0

# Run Streamlit app
echo "Starting Swedish Audio Translator..."
echo "Environment configured for macOS compatibility"
streamlit run main.py
