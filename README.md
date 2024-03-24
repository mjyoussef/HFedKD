## Overview
`HFedKD` is a meta-algorithm for model-**H**eteregenous **Fed**erated learning (FL) using **K**nowledge **D**istillation. It is designed to handles two key constraints while incurring minimal communication cost:
1. The distribution of training samples across clients is non-iid and imbalanced (ie. some clients may have more samples than others.)
2. Each client may have a different variation of a base model. For example, different clients may be training different versions of the VGG model.

We also provide implementations for isolated and clustered federated learning as baselines for comparison.