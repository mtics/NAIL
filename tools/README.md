# README

> This folder contains some tool code.

## Introduction

- To transfer raw formats to `.mat`, please run `data_read.m`.
  - Before run it, you should change the `loadPath` and `savePath` to where you put raw datasets and where you want to store the transferred datasets.
  - The datasets supported:
    - 'corel5k', 'mirflickr', 'pascal07', 'iaprtc12'
      - supported by `vec*.m`.
    - 'emotions', 'yeast'
      - supported by `arff2mat.m`.
- Use `data_load()` to load your dataset.
- Use `data_merge()` to merge the train set and the test set.
- Use `data_split()` to randomly divide the dataset into train set and test set according to a rate.
- Use `data_simulation()` to randomly generate data matrices that meets our assumptions.
- `mask()` will generate masked data matrix, and their indicator matrix.
- We use `maxide`[1] to handle the optimization w.r.t the body label mapping matrices.

## Reference

[1] Miao Xu, Rong Jin, Zhi-Hua Zhou. **Speedup Matrix Completion with Side Information: Application to Multi-Label Learning**. In: Advances in Neural Information Processing Systems 26 (NIPS 2013), Lake Tahoe, 2013.
