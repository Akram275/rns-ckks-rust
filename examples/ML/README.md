# ML Examples

This directory contains simple machine-learning-oriented CKKS inference examples.

- `ml_linear_regression.rs`: encrypted linear regression inference with public weights.
- `ml_logistic_regression.rs`: encrypted logistic regression inference using a Paterson-Stockmeyer polynomial approximation of the sigmoid.

These examples keep model weights in plaintext and encrypt only the input feature vector.