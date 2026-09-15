# SGNav — Semantic-Guided USV Navigation

[![arXiv](https://img.shields.io/badge/arXiv-2609.14558-b31b1b.svg)](https://arxiv.org/abs/2609.14558)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2609.14558-blue)](https://doi.org/10.48550/arXiv.2609.14558)

A lightweight public release of **SGNav**, a semantic-guided navigation framework for autonomous surface vessels in simulated harbor environments.

---

## Research Overview

SGNav extends RL-based navigation with **semantic visual perception**, enabling the agent to identify language-specified targets and navigate toward them in visually complex environments.

- **Task:** Semantic goal-directed USV navigation
- **Perception:** GroundingDINO + CLIP-based semantic localization
- **Control:** Learned navigation policy with visual/semantic guidance
- **Environment:** Unity harbor simulation with representative Task-1 evaluation

This release includes one representative Unity scene, a trained checkpoint, evaluation code, and runtime profiling tools.

### Overall Architecture

<p align="center">
  <a href="https://github.com/user-attachments/assets/009b67c3-766f-438e-8211-1b95c661c182">
    <img
      src="https://github.com/user-attachments/assets/009b67c3-766f-438e-8211-1b95c661c182"
      width="620"
      alt="SGNav Overall Architecture"
    />
  </a>
</p>

The framework connects semantic perception, target localization, and policy-based navigation in a closed-loop agent pipeline.

---

## Representative Results

Representative qualitative and quantitative results from the SGNav evaluation:

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/4bb8a481-2f05-45d3-a29f-c78cc334eb9b"
    width="620"
    alt="SGNav Representative Results"
  />
</p>

For the complete experimental setup, ablations, and evaluation results, please refer to the paper.

---

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate usv-clip
