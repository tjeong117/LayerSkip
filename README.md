<p align="center">
<img width="200px" alt="OpenMoE" src="https://github.com/XueFuzhao/OpenMoE/blob/main/logo.jpg?raw=true">
</p>
<p align="center"><a href="https://github.com/XueFuzhao/OpenMoE/tree/main">[Homepage]</a> | <a href="https://arxiv.org/abs/2402.01739">[Paper]</a> |  <a href="https://colab.research.google.com/drive/1xIfIVafnlCP2XVICmRwkUFK3cwTJYjCY#scrollTo=62T-2mH_tsjG">[Colab Demo]</a> | <a href="https://huggingface.co/OrionZheng">[Huggingface]</a> | <a href="https://discord.gg/bjGnGfjegU">[Discord]</a>  |  <a href="https://twitter.com/xuefz/status/1693696988611739947?s=61&t=Xc2k2W7vU_hlpNizGDCmOw">[Twitter]</a> | <a href="https://xuefuzhao.notion.site/Aug-2023-OpenMoE-v0-2-Release-43808efc0f5845caa788f2db52021879">[Blog]</a></p>
</p>
<hr>

# OpenMoE with LayerSkip
OpenMoE is a project aimed at igniting the open-source MoE community! We are releasing a family of open-sourced Mixture-of-Experts (MoE) Large Language Models.

## Proposed Feature: LayerSkip for MoE
We're excited to propose **LayerSkip for MoE**, a novel technique that aims to enhance our existing Mixture of Experts architecture by dynamically bypassing layers based on input complexity. This project is currently in development as part of our research initiative. The proposed approach offers several potential benefits:

- **Improved Computational Efficiency**: Selectively processing inputs through fewer layers when full depth isn't needed
- **Enhanced Expert Utilization**: Better load balancing across experts through dynamic routing
- **Faster Inference**: Integration with speculative decoding specifically optimized for MoE architectures
- **Early Exit Capabilities**: Confidence-based termination of processing when sufficient output quality is achieved

### Planned LayerSkip Implementation
Our planned implementation will focus on:
1. **Expert Dropout during Training**: Random deactivation of experts to improve robustness
2. **Intermediate Output Layers**: Added between expert groups to enable early exit
3. **Confidence-based Exit Mechanism**: Dynamic layer skipping based on prediction confidence
4. **Integration with Speculative Decoding**: Using a smaller draft model to propose candidate tokens

## Project Contributors & Course Information
This project is being developed as part of **Georgia Tech's Advanced Deep Learning Course (CSE 8803/CS 7643)** for Spring 2025.

### Team Members:
- Nicholas Papciak 
- Kavya Golamaru
- Vishal Maradana
- Rishi Bandi

## How to Contribute
We're actively seeking contributors to help implement LayerSkip for MoE. Here are the key areas where we need assistance:

### Implementation Needs:
1. **Expert Dropout Mechanism**: 
   - Implement random deactivation of experts during training
   - Modify the gating network to redistribute probabilities among active experts

2. **Early Exit Framework**:
   - Design and implement intermediate output layers after groups of experts
   - Develop loss computation at each intermediate layer
   - Create confidence-based early exit mechanisms

3. **Speculative Decoding Integration**:
   - Implement a draft model system for token proposal
   - Develop verification mechanisms for proposed tokens
   - Optimize the interaction between speculative decoding and MoE routing

4. **Testing and Evaluation**:
   - Design benchmark tests to evaluate performance improvements
   - Create metrics for measuring expert utilization efficiency
   - Compare inference speed against baseline MoE models

If you're interested in contributing to any of these areas, please reach out to the team members or open an issue in this repository.

## News

[2025/02] 🔥 **Project Proposal**: We're developing LayerSkip for MoE, a novel technique that aims to dynamically adjust processing depth, improving computational efficiency and expert utilization!

