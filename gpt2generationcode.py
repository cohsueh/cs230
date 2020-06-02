! pip install transformers

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, LineByLineTextDataset, ... 
                         DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch


config = GPT2Config()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_tokens('<pad>')
tokenizer.pad_token = '<pad>'

model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="all.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./GPT2-2",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_gpu_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

prompt = '''JOHNNY
Well because it was an out of state bank. Anyway, I was working as a busboy in a hotel, and she was sitting, drinking her coffee, and she was so beautiful, and I say hi to her. That’s how we met.

MARK
So, I mean, what's the interesting part?

JOHNNY
Well the interesting part is that
'''
inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
outputs = model.generate(inputs, max_length=300, do_sample=True, top_p=0.95, top_k=100, temperature=1.1)

tokenizer.decode(outputs[0].numpy())