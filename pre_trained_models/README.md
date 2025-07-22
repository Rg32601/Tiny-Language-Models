# Pretrained BERT6 Model
**Tag:** pretrain-tinybert-layers6-conv0-k3-d3-maxlen128-vocab30522-ep50-wikisize90000 
**Date:** 2024-07-22

- Layers: 6
- Conv layers: 0
- Kernel size: -
- D: -
- Max length: 128
- Vocab size: 30522
- Wiki size: 90000
- Pretrain epochs: 50

## Usage
```python
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained("pretrain-tinybert-layers6-conv0-k3-d3-maxlen128-vocab30522-ep50-wikisize90000")
