# lm

A small (enough) transformer model trained on tiny shakespeare dataset.

![Screenshot 2025-01-11 at 08.44.46.png](screenshots/Screenshot%202025-01-11%20at%2008.44.46.png)

Text samples:

```commandline
step=0: ['\nscncn0L\'ng\x03\x04ZA-VuN7&\x07\x0cp0Pa\x1be7\\ZQ\r]P\x0f\x05QwK\x0fM^4lW\x08c\x06Z\x1f\x11\ns\x19<\tt\x1bes, Ue,%\x01/0LLc0?2\x07Rsc0L/m0Ue7\\Du\x03$"7j\x13is ']

step=500: ["\nSI'sl oll hat bes the, wald mash hasesthe ine, maime.\n\nBRUS:\nIt it shos she trow sun mare thaly she "]

step=1000: ['\n\nI than wither be her and her sporand of mughthing.\n\nCiraienit shincederer oftoor shrem mas butst an']

step=2000: ['\nPromostent, there saide, I sakes be the,-\n\nCAPULET:\nMay grace:\nHe to the thmelf that herefull bovest']

step=2500: ['\nAst that would whendile thy what to war hence here to shest\nAnd bllesst asted, heavance and his tran']

step=3000: ["\n\n\nANGETIZALO:\nNo, sir! I not amore him and foe it the sorows.\n\nPEGRETIII:\nWending o't the contUponsi"]

step=3500: ['\n\nPent:\nWithing sinsteed is hand shem thither show a subjected\nAnd hand the care terurn it the town h']

step=4000: ['\nThe belieness is pleop of thyself woome offecte to man\nastie thy heaven as imwer in the doth say the']

step=4500: ['\nWilit we well.\n\nBRUTUS:\nIs it strance that have see higents again?\n\nGREY:\nHow save a thou art our be']

step=5000: ['\n\nLord March! Come not the comfort and at tell born a please in.\nHow harm. What it thou doest thus se']
```

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