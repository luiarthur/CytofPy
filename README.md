# CytofPy
Implementation of Cytof model using Automatic Differentiation Variational
Inference (ADVI) and PyTorch.

# References

- [ADVI][3]
- [Variational Inference][4]
- [Stochastic Variational Inference][5]
- [Unit Simplex (Stick Breaking) Transform][6]
- ~~Taking gradients w.r.t. parameters of random variates~~
    - ~~[Pathwise Derivatives Beyond the Reparameterization Trick][1]~~
    - ~~[PyTorch `rsample`][2]~~

[1]: https://arxiv.org/pdf/1806.01851.pdf
[2]: https://pytorch.org/docs/stable/distributions.html#gamma
[3]: http://jmlr.org/papers/volume18/16-107/16-107.pdf
[4]: https://arxiv.org/pdf/1601.00670.pdf
[5]: https://arxiv.org/pdf/1206.7051.pdf
[6]: https://mc-stan.org/docs/2_18/reference-manual/simplex-transform-section.html
