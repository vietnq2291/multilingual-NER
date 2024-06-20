from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    MT5TokenizerFast,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)
from train.utils import *
import wandb

# Initialization
wandb.init(
    project='thesis',
    name='mt0_base-sft-open_ner_en_only',
)

output_model_dir = 'mt0_base-sft-open_ner_en_only'
model_checkpoint = "bigscience/mt0-base"


def main():
    # Load model and tokenizer
    tokenizer = MT5TokenizerFast.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Prepare dataset
    dataset = load_dataset('nqv2291/en-mT5-ner-open_domain')

    # Prepare training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        padding='longest',
        return_tensors="pt",
        pad_to_multiple_of=8
    )

    args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,
        overwrite_output_dir=True,
        report_to='wandb',
        save_strategy = "epoch",
        eval_strategy = "steps",
        num_train_epochs=3,
        eval_steps=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=64,
        gradient_checkpointing=True,
        logging_steps=10,
        learning_rate=1e-4,
        lr_scheduler_type='constant',
        # predict_with_generate=True,
        bf16=True,
        tf32=False,
        optim='adafactor',
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=lambda x: compute_bleu(x, tokenizer),
    )

    model.train()
    trainer.train()
    print("Training finished!")

if __name__ == '__main__':
    main()
