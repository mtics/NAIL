# NAIL

> This project is the code and the supplementary of "**Incomplete multi-view weak-label learning with noisy features and imbalanced labels**"


## Requirements

- **Matlab** (the version we used: *R2020b*)
- Some matlab toolboxes:
  - **Matlab Weka Interface** (to use their functions `loadARFF()` and `weka2matlab()` in `arff2mat()`)
  - **Statistics and Machine Learning Toolbox** (to use their function in `zscore()` in `data_load()`)

## Quick Start

To run our method, please run:

```matlab
run_alg('nail', 'linear', 'emotions', False, lambda, mu, subRatio, 1, 0.5, 0.5)
```

where `lambda`, `mu` and `subRatio` are three hyper-parameters.

## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))
