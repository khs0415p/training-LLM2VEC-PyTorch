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

## Start

- Masked Next Token Prediction

```
python train_mntp.py --model_name [model name or path]
```

- SimCSE
```
python train_mntp.py --model_name [model name or path] --peft_model_name_or_path [peft name or path]
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