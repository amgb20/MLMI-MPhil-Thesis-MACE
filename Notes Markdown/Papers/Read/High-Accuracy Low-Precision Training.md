# High-Accuracy Low-Precision Training


### 📘 1. Overview of HALP

* HALP is a **low‑precision SGD variant** that achieves **high accuracy** by combining:

  1. **SVRG** (Stochastic Variance-Reduced Gradient) to suppress gradient noise.
  2. **Bit-centering**, a technique that periodically re-centers the low-precision representation using a higher-precision "offset" from an SVRG full-gradient pass.
     This keep quantization error in-check.
  3. **Dynamic bias adjustment** to prevent overflow/underflow ([cs.cornell.edu][1], [dawn.cs.stanford.edu][2]).

* On strongly convex problems, HALP provably **converges at the same linear rate** as full-precision SVRG and, importantly, **achieves arbitrarily accurate solutions** despite using fixed-bit low precision ([cs.cornell.edu][1]).

* In practical tests (including CNNs and LSTMs), HALP with 8-bit precision matches the validation performance of full-precision training, outperforming plain low-precision SGD ([arxiv.org][3]).

---

### 🎯 2. Main Findings

* **Bit‑centering** is key: by keeping most updates in low precision but recentering using full-precision offsets, HALP reduces quantization noise as training progresses ([cs.cornell.edu][1]).
* **SVRG integration** ensures that gradient variance doesn’t blow up low-precision convergence.
* **Theory + practice**: Both theoretical guarantees (strongly convex problems) and empirical validations (resnet, LSTM) demonstrate HALP’s effectiveness ([arxiv.org][3], [cs.cornell.edu][1]).
* Performance-wise, it achieved up to **4× speed-up** over full-precision SVRG on CPU and matched convergence ([arxiv.org][3]).

