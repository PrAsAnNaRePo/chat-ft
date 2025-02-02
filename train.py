import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
import wandb
from trl import SFTTrainer
from process_data import ProcessData
import argparse
from unsloth import FastLanguageModel

def train(args):

    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds = ProcessData(args.chat_file_path, tokenizer, args.hr_gap).get_rc_msg_format(target_role=args.target_role)

    print("Lenght of dataset: ", len(ds))

    if args.low_gpu_memory:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            max_seq_length = args.max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            max_seq_length = args.max_seq_length,
            use_rslora = False,
            loftq_config = None,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            tokenizer.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing
        
        # Tokenize the texts and process the images
        batch = tokenizer(
            texts, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors
    
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        batch["labels"] = labels  # Add labels to the batch
    
        return batch  # Return the prepared batch

    # print(dir(SFTConfig))
    
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=args.output_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="steps",
        save_steps=20,
        bf16=args.training_dtype == "bf16",
        fp16=args.training_dtype == "fp16",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=True,
        report_to="wandb" if args.wandb_logger else None,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length,
    )
    
    print("Number of GPUs: ", training_args.n_gpu)

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    if args.wandb_logger:
        wandb.init(
            project="chat-ft",  # change this
            name="qwen2.5-2b-chat-ft",  # change this
            config=training_args,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HF Model ID")
    parser.add_argument("--chat_file_path", type=str, required=True, help="Path to the chat file")
    parser.add_argument("--hr_gap", type=int, default=5, help="hour gap for grouping messages")
    parser.add_argument("--target_role", type=str, required=True, help="Target role for the conversation to be trained on")
    parser.add_argument("--low_gpu_memory", type=bool, default=True, help="Uses pfet to lower gpu memory")
    parser.add_argument("--training_dtype", type=str, default="fp16", help="Training dtype")
    parser.add_argument("--output_path", type=str, default="qwen2.5-1.5b-chat-ft", help="Path to save the model")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--wandb_logger", type=bool, default=True, help="Use wandb for logging")
    parser.add_argument("--max_seq_length", type=int, default=3000, help="Maximum sequence length for training")
    args = parser.parse_args()

    train(args)