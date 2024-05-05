from datasets import load_dataset

dataset = load_dataset("./dataset")
dataset.push_to_hub("frank")

imdb = load_dataset("frankie699/frank")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
# print(tokenized_imdb["test"][0])

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np

from sklearn import metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy":accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        "macro_f1":metrics.f1_score(labels, predictions, average='macro')}

id2label = {0: "one", 1: "two"}
label2id = {"one": 0, "two": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v2-xxlarge", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir='frankie699/final_project_output',
    overwrite_output_dir = True,
    learning_rate=3e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    logging_dir='frankie699/output',
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    push_to_hub=True,
    fp16=True,
    gradient_checkpointing=True,
    metric_for_best_model="eval_macro_f1",
    optim="adamw_bnb_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.push_to_hub()