---# Mine Reclamation AI Framework

**Author:** Daniel Agyekum Amakye
**Date:** 2025-05-24

## Overview

Three-phase AI framework for adaptive mine reclamation.

### Phase 1: KED-Transformer Fusion
- Fuses 11 modalities into unified state representation
- Output: 100x100x11 grid

### Phase 2: GNN-LSTM Prediction
- Predicts pH recovery trajectories
- RMSE: 0.3875

### Phase 3: DRL Control
- Soft Actor-Critic algorithm
- 37.8 percent improvement over baseline
- Final pH: 6.13 (meets regulatory standard)

## Key Results

| Metric | Value |
|--------|-------|
| Improvement | 37.8 percent |
| Final pH DRL | 6.13 |
| Final pH Baseline | 4.98 |
| Features Fused | 11 |

## Installation

```bash
pip install -r requirements.txt