from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    MT5TokenizerFast,
    MT5ForConditionalGeneration, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)
from train.utils import *
import wandb

# Initialization
wandb.init(
    project='thesis',
    name='mt5_large-sft-en_vi_alpaca',
)

output_model_dir = 'mt5_large-sft-en_vi_alpaca'
model_checkpoint = "google/mt5-large"
test_size = 0.1
seed = 42


def main():
    # Load model and tokenizer
    tokenizer = MT5TokenizerFast.from_pretrained(model_checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # Prepare dataset
    vi_alpaca = load_dataset('nqv2291/vi-mT5-instruction_tuning-bkai', split='train')
    en_alpaca = load_dataset('nqv2291/en-mT5-instruction_tuning-gpt4', split='train')

    vi_alpaca = vi_alpaca.train_test_split(test_size=test_size)
    en_alpaca = en_alpaca.train_test_split(test_size=test_size)

    dataset = DatasetDict({
        'train': concatenate_datasets([vi_alpaca['train'], en_alpaca['train']]),
        'val': concatenate_datasets([vi_alpaca['test'], en_alpaca['test']])
    })
    dataset.shuffle(seed)

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
        save_strategy = "steps",
        eval_strategy = "steps",
        num_train_epochs=3,
        save_steps=500,
        eval_steps=250,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=64,
        gradient_checkpointing=True,
        logging_steps=10,
        learning_rate=1e-4,
        lr_scheduler_type='constant',
        predict_with_generate=True,
        bf16=True,
        tf32=False,
        optim='adafactor',
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_rouge(x, tokenizer),
    )

    model.train()
    trainer.train()
    print("Training finished!")

if __name__ == '__main__':
    main()
