
# BERT Fine-Tuning for Named Entity Recognition (NER)

 Overview
Named Entity Recognition (NER) is a crucial task in Natural Language Processing (NLP) that involves identifying and classifying entities in text into predefined categories such as names, locations, organizations, and more. This project fine-tunes a BERT (Bidirectional Encoder Representations from Transformers) model for NER using the CoNLL-2003 dataset. 

BERT, introduced by Google, is a transformer-based model that utilizes self-attention mechanisms and bidirectional context learning, making it highly effective for token-level classification tasks like NER.

Dataset: CoNLL-2003
The **CoNLL-2003 dataset** is widely used for training and evaluating NER models. It consists of news articles annotated with entity tags. The dataset includes:
- Token: Words in sentences.
- NER tags: Labels corresponding to each token.
- POS tags: Part-of-speech information.
- Chunk tags: Syntactic chunking labels.
  
For example
Entities are classified into four main categories:
- PER → Person (e.g., "Elon Musk")
- LOC → Location (e.g., "India", "New York")
- ORG → Organization (e.g., "Google", "NASA")
- MISC → Miscellaneous (e.g., "Olympics", "iPhone")

## Project Workflow

1. Install Dependencies
Run the following command to install the required libraries:
```
pip install -q transformers[torch] datasets accelerate tokenizers seqeval evaluate
```

 2. Load the Dataset
```python
import datasets
conll2003 = datasets.load_dataset("conll2003")
```

3. Tokenization
We use **BERT tokenizer** to process the text:
```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
example_text = conll2003['train'][0]
tokenized_id = tokenizer(example_text["tokens"], is_split_into_words=True)
```

 4. Model Setup
BERT is pre-trained on vast amounts of text data, making it an excellent choice for transfer learning in NLP. We fine-tune a **pretrained BERT model for token classification:
```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=9  # 9 includes O and entity labels
)
```

 5. Training
Fine-tuning involves adjusting the pre-trained model weights to optimize performance on the CoNLL-2003 dataset. This is done using **Hugging Face's Trainer API**.
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert_ner", per_device_train_batch_size=8, num_train_epochs=3
)
trainer = Trainer(model=model, args=training_args, train_dataset=conll2003["train"])

trainer.train()
```

 6. Evaluation
After training, the model is evaluated using standard NER metrics such as **precision, recall, and F1-score**.
```python
trainer.evaluate()
```


7. How to Use the Model
Once trained, the model can be used for entity recognition on custom text.
```python
from transformers import pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
text = "Elon Musk is the CEO of Tesla, headquartered in California."
print(ner_pipeline(text))
```


## Credits
- Dataset: [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- Transformers Library: [Hugging Face](https://huggingface.co/transformers/)
- Trainer API: Hugging Face

GitHub Repository

The fine-tuned model has been uploaded to GitHub. You can find it here: sourabhmatali/NER-Model-Fine-Tuned

