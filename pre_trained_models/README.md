# Pretrained TinyBERT Model
**Tag:** v1.0  
**Date:** 2024-07-22

- Layers: 6
- Conv layers: 2
- Kernel size: 3
- D: 64
- Max length: 128
- Vocab size: 30522
- Wiki size: 100000
- Pretrain epochs: 10
- Learning rate: 0.0001

## Usage
```python
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained("pretrain-tinybert-layers6-conv2-k3-d64-...")
