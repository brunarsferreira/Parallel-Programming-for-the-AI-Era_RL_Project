#!/bin/bash
# Setup script for Pendulum RL project on Grid5000
# Creates a virtual environment and installs all dependencies

echo "=== Setting up Pendulum RL Environment ==="

# Create virtual environment
python3 -m venv pendulum_venv

# Activate virtual environment
source pendulum_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Ray with RLlib support and PyTorch
pip install "ray[rllib]" torch

# Install Gymnasium with classic control environments
pip install "gymnasium[classic_control]"

# Additional required packages (pydantic is needed by ray.train)
pip install numpy pandas matplotlib pydantic

echo "=== Installation Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source pendulum_venv/bin/activate"
echo ""
echo "To test the installation:"
echo "  python -c \"import ray; import gymnasium; print('Ray version:', ray.__version__); print('Gymnasium version:', gymnasium.__version__)\""
