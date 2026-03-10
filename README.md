# TensorEstimation
These codes simulate a model CP tensor + noise and estimate it via ALS.

In this project I simulate tensors as the sum of a rank $R$ CP tensor and a noise tensor. I let the order $d$ and the rank $R$ vary, in particular $d\in[3,4,5]$ and $R\in[3,4,5]$. The CP tensor is constructed in such a way to have a prescribed level of column dependence $\xi$. I then estimate these tensors via the Alternating Least Squares (ALS) algorithm and I observe, for each simulation, the number of iterations needed to the algorithm to reach a specific level of the error metric $\varepsilon$. For different levels of the controlled coherence parameter $\xi$ and of the signal-to-noise ratio the algorithm behaves differently.
