# Multilingual Named Entity Recognition for Indian Languages

A comparative multilingual NER project for **Hindi, Urdu, Odia, and Telugu**. The project investigates whether joint multilingual training improves named entity recognition and compares an Indian-language-specialized representation model with a general multilingual transformer.

> **Dataset source and reference:** This project uses the four-language annotated corpus presented in Bahad et al., *Multilingual Named Entity Recognition for Indian Languages*. Read the paper: [arXiv:2405.04829](https://arxiv.org/pdf/2405.04829). The paper presents an approximately 40K-sentence corpus for Hindi, Urdu, Odia, and Telugu across Indo-Aryan and Dravidian language families. [web:1]

## Project Highlights

- Trained on **34,898 sentences** and evaluated on separate test sets for four Indian languages.
- Used a shared **13-label BIO tag scheme** covering six entity categories: `NEAR`, `NEL`, `NEN`, `NEO`, `NEP`, and `NETI`.
- Compared two multilingual architectures:
  - **Tier 1:** Frozen MuRIL embeddings + BiLSTM-CRF.
  - **Tier 3:** XLM-RoBERTa + LoRA fine-tuning.
- Compared multilingual joint training with a Hindi-only baseline.
- Used language-specific test evaluation so that transfer performance could be analyzed separately.

## Dataset

The project uses CoNLL-style token-level NER data for four languages:

| Language | Train | Dev | Test |
|---|---:|---:|---:|
| Hindi | 11,076 | 1,389 | 1,388 |
| Urdu | 8,720 | 1,094 | 1,096 |
| Odia | 12,109 | 1,517 | 1,519 |
| Telugu | 2,993 | 384 | 384 |
| **Total** | **34,898** | **4,384** | **4,387** |

The test sets remain separate and are never merged into the training pool.

## Models

### Tier 1: MuRIL + BiLSTM-CRF

MuRIL is used as a frozen encoder to generate contextual word-level embeddings. A bidirectional LSTM models sequence context, and a CRF layer enforces valid tag transitions.

```text
Input words
    ↓
Frozen google/muril-base-cased
    ↓
Word-level subword-pooled embeddings
    ↓
BiLSTM
    ↓
CRF decoder
    ↓
BIO entity tags
```

### Tier 3: XLM-RoBERTa + LoRA

XLM-RoBERTa is adapted using parameter-efficient LoRA fine-tuning. The model uses token-level classification with BIO labels and ignores special-token positions during loss and evaluation.

```text
Input words
    ↓
XLM-RoBERTa tokenizer
    ↓
XLM-RoBERTa + LoRA adapters
    ↓
Token classification head
    ↓
BIO entity tags
```

## Results

### Multilingual Tier 1 vs. Tier 3

| Language | Tier 1: MuRIL + BiLSTM-CRF | Tier 3: XLM-R + LoRA |
|---|---:|---:|
| Hindi | **0.8429** | 0.8077 |
| Urdu | **0.8293** | 0.7866 |
| Odia | 0.6701 | **0.6953** |
| Telugu | 0.7508 | **0.7653** |
| Combined dev F1 | **0.8136** | 0.7956 |

The best overall model is the multilingual Tier 1 MuRIL + BiLSTM-CRF system, although Tier 3 performs better on Odia and Telugu.

### Hindi: Monolingual vs. Multilingual

| Training setup | Hindi test F1 |
|---|---:|
| Hindi-only MuRIL + BiLSTM-CRF | 0.8212 |
| Multilingual MuRIL + BiLSTM-CRF | **0.8429** |

Multilingual training improved Hindi by **2.17 F1 points**. The largest class-level improvement was observed for `NEAR`, whose F1 increased from approximately **0.39 to 0.56**.

## Interpretation

- MuRIL + BiLSTM-CRF performs best on Hindi and Urdu, suggesting that Indian-language-specialized pretraining is valuable for these languages.
- XLM-R + LoRA provides modest gains on Odia and Telugu, showing that model choice can interact with language and script.
- Rare entity categories remain difficult, especially when test support is small.
- Odia numeric entities (`NEN`) are particularly challenging and require additional data and error analysis.
- Joint multilingual training can improve a target language by providing more diverse entity-boundary and entity-type examples.

## Repository Files

### `Multilingual_lstmcrf.ipynb`

Main multilingual notebook containing the combined-data pipeline and both multilingual model experiments:

- Dataset parsing and validation.
- Shared tag mapping creation.
- Tier 1 MuRIL embedding extraction.
- Tier 1 BiLSTM-CRF training and per-language evaluation.
- Tier 3 XLM-R + LoRA training and per-language evaluation.
- Result summaries and saved model artifacts.

### `NER_Hindi`

Hindi NER notebook or source file containing the Hindi-specific baseline experiment.

### `NER_Hindi_lora.ipynb`

Hindi-only XLM-R + LoRA experiment.

### `ner-hindi.ipynb`

Additional Hindi NER experiment or earlier implementation.

## Reproducibility

Install the main dependencies:

```bash
pip install torch transformers peft torchcrf seqeval numpy scikit-learn
```

The notebooks are intended to run on a CUDA-enabled environment. The base models are downloaded from Hugging Face:

- `google/muril-base-cased`
- `xlm-roberta-base`

For the multilingual run, place the dataset under the expected Kaggle path or update `BASE_PATH` in the notebook:

```python
BASE_PATH = "/kaggle/input/datasets/charumittalma25m008/ner-models-for-indian-languages/Datasets"
```

The expected directory structure is:

```text
Datasets/
├── Hindi/
│   ├── Hindi-train.txt
│   ├── Hindi-dev.txt
│   └── Hindi-test.txt
├── Urdu/
│   ├── Urdu-train.txt
│   ├── Urdu-dev.txt
│   └── Urdu-test.txt
├── Odia/
│   ├── Odia-train.txt
│   ├── Odia-dev.txt
│   └── Odia-test.txt
└── Telugu/
    ├── Telugu-train.txt
    ├── Telugu-dev.txt
    └── Telugu-test.txt
```

## Evaluation

Evaluation uses entity-level micro F1 through `seqeval`. Results are reported separately for each language. Because the datasets contain different numbers of examples per entity category, macro and weighted scores should also be inspected when analyzing rare classes.

## Limitations and Future Work

- The multilingual training pool is imbalanced, with Telugu having substantially fewer training sentences than Hindi and Odia.
- Class-level comparisons can be unstable for categories with low support.
- Odia `NEN` errors require targeted data and annotation analysis.
- Future experiments could include multilingual sampling or upsampling, MuRIL + LoRA, confidence calibration, and a lightweight inference demo.

## Reference

```bibtex
@article{bahad2024multilingual,
  title={Multilingual Named Entity Recognition for Indian Languages},
  author={Bahad, Sankalp and Mishra, Pruthwik and Arora, Karunesh and Balabantaray, Rakesh Chandra and Sharma, Dipti Misra and Krishnamurthy, Parameswari},
  journal={arXiv preprint arXiv:2405.04829},
  year={2024}
}
```

Paper: [https://arxiv.org/pdf/2405.04829](https://arxiv.org/pdf/2405.04829)

## Technologies

Python · PyTorch · Hugging Face Transformers · MuRIL · XLM-RoBERTa · LoRA · PEFT · BiLSTM · CRF · seqeval · NumPy · CUDA · Jupyter

## Author

**Charu Mittal**
