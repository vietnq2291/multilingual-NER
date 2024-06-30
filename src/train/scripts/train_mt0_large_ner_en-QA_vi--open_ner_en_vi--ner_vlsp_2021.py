from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    MT5TokenizerFast,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)

import wandb
from huggingface_hub import login as hf_login
# Initialization
hf_login(token="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")
wandb.login(relogin=True, key="4b032b8bbff0e916bbc330ca43341be268d70498")
wandb.init(
    project='thesis',
    name="nqv2291/mt0_large-sft-open_ner_en-mix-QA_vi__open_ner_en_vi__vlsp_2021"
)

output_model_dir = 'mt0_large-sft-open_ner_en-mix-QA_vi__open_ner_en_vi__vlsp_2021'
model_checkpoint = "nqv2291/mt0_large-sft-open_ner_en-mix-QA_vi__open_ner_en_vi"


def main():
    # Load model and tokenizer
    tokenizer = MT5TokenizerFast.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Prepare dataset
    dataset = load_dataset('nqv2291/vi-mT5-ner-vlsp-GOLD')
    dataset = dataset.shuffle(9)

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
        eval_steps=0.2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        gradient_checkpointing=True,
        logging_steps=1,
        learning_rate=3e-5,
        lr_scheduler_type='constant',
        # predict_with_generate=True,
        bf16=False,
        tf32=False,
        optim='adafactor',
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=lambda x: compute_bleu(x, tokenizer),
    )

    model.train()
    trainer.train()
    print("Training finished!")

if __name__ == '__main__':
    main()