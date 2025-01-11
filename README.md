# lm

A small (enough) transformer model trained on tiny shakespeare dataset.

![Screenshot 2025-01-11 at 08.44.46.png](screenshots/Screenshot%202025-01-11%20at%2008.44.46.png)

Goals:
* Understand the minimum configurations that can make an autoregressive LM "work".

Implementations:
* A simple ASCII tokenizer using python's `ord()` and `chr()`.
* Top K sampling

Notes:
* Does training loss decrease and converge means a correct implementation or successful model?
  * No! There was [a bug in calculating attention mask](https://github.com/k-duan/lm/commit/38e208e91d187986119ad290c85940f88d07c67a), but the LM can still overfit training data with that bug. With that bug, the text generated by the model is completely nonsense.
* Is position embedding really necessary?
  * No! It's not mandatory for an autoregressive model to work, *if causal mask is used*.
* Most commonly made implementation mistakes
  * Must use `-inf` instead of `0` to mask logits, e.g. [attention mask](https://github.com/k-duan/lm/commit/38e208e91d187986119ad290c85940f88d07c67a), [top k sampling](https://github.com/k-duan/lm/commit/fbf2561cabac7b92c1e7c53e0db740c49ab30769).

References:
* The code is written from scratch but several bugs were found when comparing with Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) repo.