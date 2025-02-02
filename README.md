# Chat-ft

This project fine-tunes the any llm model on WhatsApp chat logs using the Hugging Face Transformers library. The goal is to adapt the model to generate conversational responses in a style similar to the chat logs. The project includes data preprocessing, token distribution analysis, and fine-tuning with LoRA (Low-Rank Adaptation) for efficient training on low-memory GPUs.


## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Data Preprocessing](#data-preprocessing)
6. [Training](#training)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Features
- **Data Preprocessing**: Parses WhatsApp chat logs, combines consecutive messages, and groups conversations based on time gaps.
- **Token Distribution Analysis**: Visualizes token distribution across messages and users.
- **LoRA Fine-Tuning**: Efficiently fine-tunes the model using LoRA for low-memory GPUs.
- **WandB Integration**: Logs training metrics and hyperparameters using Weights & Biases.

## Requirements
- Python 3.8+
- Libraries:
  - `transformers`
  - `torch`
  - `peft`
  - `trl`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `wandb`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PrAsAnNaRePo/chat-ft.git
   cd whatsapp-chat-finetuning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Preprocessing
The `ProcessData` class in `process_data.py` handles the preprocessing of WhatsApp chat logs. It:
- Parses chat logs into structured data.
- Combines consecutive messages from the same user.
- Groups messages into conversations based on a time gap.
- Visualizes token distribution and message statistics.

### Training
The `train.py` script fine-tunes the Qwen2.5 model on the preprocessed chat logs. It supports:
- LoRA for low-memory GPUs.
- Customizable training parameters (batch size, learning rate, etc.).
- WandB logging for tracking training progress.

## Data Preprocessing
### Input Format
- WhatsApp chat logs should be in a text file with the following format:
  ```
  dd/mm/yy, hh:mm am/pm - Name: Message
  ```
- Example:
  ```
  21/03/24, 7:40 pm - John: Hello
  21/03/24, 7:41 pm - Jane: Hi there!
  ```

### Preprocessing Steps
1. Parse chat logs into structured data.
2. Combine consecutive messages from the same user.
3. Group messages into conversations based on a time gap (default: 5 hours).

### Token Distribution Analysis
- **Histogram**: Overall token distribution per message.
- **Box Plot**: Token distribution by user.
- **Time Series**: Token counts over time.

> Note: also feel free to change the default system prompt at `process_data.py` in line `316` and play around with time gap period in `group_messages` function in `ProcessData`.

## Training
### Command
```bash
python train.py \
  --chat_file_path chat_file.txt \
  --target_role "John" \
  --low_gpu_memory True \
  --output_path ./qwen2.5-1.5B-chat-ft \
  --num_epochs 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --wandb_logger True
```

### Arguments
- `--chat_file_path`: Path to the WhatsApp chat log file.
- `--target_role`: The user whose messages will be treated as the assistant's responses.
- `--low_gpu_memory`: Use LoRA for low-memory GPUs (default: `True`).
- `--output_path`: Directory to save the fine-tuned model.
- `--num_epochs`: Number of training epochs (default: `4`).
- `--per_device_train_batch_size`: Batch size per device (default: `4`).
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `8`).
- `--wandb_logger`: Enable WandB logging (default: `True`).

## Results
- The fine-tuned model will be saved in the specified `output_path`.
- Training metrics (loss, learning rate, etc.) will be logged to WandB (if enabled).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.