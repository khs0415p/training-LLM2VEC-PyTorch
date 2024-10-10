# LLM2VEC

LLM2VEC에 대한 설명은 [여기](https://velog.io/@khs0415p/Paper-LLM2Vec)를 참고하세요.

## Abstract

decoder-only LLM도 encoder-only 모델의 embedding 성능과 유사하거나 약간 높지만, decoder-only LLM을 bidirectional한 구조로 변경하고 Masked Next Token Prediction(MNTP)방식으로 학습시켜 word level에 대한 성능을 높이고, sentence level의 성능을 높이기 위해 simcse 학습을 수행하여 성능 향상을 이뤄냈다. 

## Dataset

MNTP와 Unsupervised SimCSE학습을 하기 위해, 문장으로 이루어진 말뭉치를 준비한다.

- wikidata

## Tree

```
.
├── data
│   └── sim_cse
│       └── data.txt                      # data for training
│
├── dataset                               # dataset class for simcse training
│   ├── __init__.py
│   ├── dataset.py
│   └── press.py
│
├── logs                                  # tensorboard folder
│
├── models                                # modeling folder
│   ├── __init__.py
│   ├── bidirectional_llama.py
│   └── bidirectional_mistral.py
│
├── results                               # saved model folder
│
├── train_mntp.py                         # run file for mntp training
│
├── train_simcse.py                       # run file for simcse training
│
└── utils                                 
    ├── __init__.py
    ├── collator.py                       # datacollator
    ├── loss_utils.py                     # contrastive loss
    ├── train_utils.py                    # util file for mistral masking
    └── trainers.py                       # trainer file
```

## Training

- Masked Next Token Prediction

```
python train_mntp.py --model_name [model name or path]
```

- SimCSE
```
python train_simcse.py --model_name [model name or path] --peft_model_name_or_path [peft name or ath]
```

## LLM2VEC

```python
import torch
from models import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

tokenzier = AutoTokenizer.from_pretrained("markim/BioLLM2VEC-Mistral-7B-v0.3-mntp", cache_dir="/data")
tokenzier.padding_side = "left"
config = AutoConfig.from_pretrained("markim/BioLLM2VEC-Mistral-7B-v0.3-mntp", cache_dir="/data")
model = AutoModel.from_pretrained("markim/BioLLM2VEC-Mistral-7B-v0.3-mntp", cache_dir="/data", torch_dtype=torch.bfloat16, device_map="cuda")

model = PeftModel.from_pretrained(model, "markim/BioLLM2VEC-Mistral-7B-v0.3-mntp", cache_dir="/data")
model = model.merge_and_unload()

model = PeftModel.from_pretrained(model, "markim/BioLLM2VEC-Mistral-7B-v0.3-simcse", cache_dir="data")

l2v = LLM2Vec(model, tokenzier)

# Encoding queries using instructions
instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "medicine to take when you have a cold"],
    [instruction, "how to cure cancer"],
]
q_reps = l2v.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "For high fever, antipyretics such as acetaminophen or aspirin can be used. For symptoms like a runny nose, nasal congestion, or sneezing, antihistamines such as chlorpheniramine, carbinoxamine, triprolidine, diphenhydramine, pseudoephedrine, and phenylephrine are helpful.",
    "The main methods for treating cancer are broadly divided into three categories: surgery, chemotherapy, and radiation therapy. In addition to these, there are other treatments such as localized therapy, hormone therapy, photodynamic therapy, and laser therapy. Recently, immunotherapy and gene therapy have also been included. Furthermore, treatments such as embolization, immunotherapy, and isotope therapy are also available.",
]
d_reps = l2v.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
"""
tensor([[0.4216, 0.3862],
        [0.2499, 0.3826]])
"""
```

## Citation

```bibtex
@article{llm2vec,
      title={{LLM2Vec}: {L}arge Language Models Are Secretly Powerful Text Encoders}, 
      author={Parishad BehnamGhader and Vaibhav Adlakha and Marius Mosbach and Dzmitry Bahdanau and Nicolas Chapados and Siva Reddy},
      year={2024},
      journal={arXiv preprint},
      url={https://arxiv.org/abs/2404.05961}
}
```