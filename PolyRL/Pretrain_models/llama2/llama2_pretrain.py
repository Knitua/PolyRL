import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from typing import List
from mytokenizers import SMILESTokenizerEnamine
from llama2 import define_llama2_configuration, Llama2
from torch.utils.data import Dataset, DataLoader, default_collate

# ====== Step 1: Load Vocabulary ======
def load_vocabulary(vocab_path: str) -> List[str]:
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab

# ====== Step 2: Custom Dataset ======
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for smiles in smiles_list:
            tokens = tokenizer.tokenize(smiles, with_begin_and_end=True)
            token_ids = [vocab.index(t) if t in vocab else vocab.index("<pad>") for t in tokens]
            # Padding or truncating
            token_ids = token_ids[:max_length] + [vocab.index("<pad>")] * max(0, max_length - len(token_ids))
            self.data.append(torch.tensor(token_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx][:-1].clone().long()
        labels = self.data[idx][1:].clone().long()
        return {"input_ids": input_ids, "labels": labels}

# ====== Step 3: LightningModule for Llama2 ======
class Llama2FineTuner(pl.LightningModule):
    def __init__(self, config, tokenizer, vocab_size, learning_rate=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.model = Llama2(config)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.index("<pad>"))

    def forward(self, input_ids):
        attention_mask = (input_ids != vocab.index("<pad>")).long()
        self.model.set_train_mode(True)
        hidden = self.model(input_ids, attention_mask)  # [batch_size, hidden_size]

        logits = self.lm_head(hidden)  # logits: [batch_size, seq_len, vocab_size]

        # 打印 logits 的形状，确认它是 [batch_size, seq_len, vocab_size]
        print(f"logits shape after lm_head: {logits.shape}")

        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]  # [batch, seq]
        labels = batch["labels"]        # [batch, seq]

        logits = self(input_ids)        # logits: [batch, seq, vocab]

        # 打印 logits 和 labels 的形状，检查是否一致
        print(f"logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
        print(f"labels shape: {labels.shape}")  # [batch_size, seq_len]

        # 展平 logits 和 labels，使它们的形状一致
        logits = logits.view(-1, logits.size(-1))  # [batch * seq, vocab]
        labels = labels.view(-1)                   # [batch * seq]

        # 再次打印展平后的形状
        print(f"flattened logits shape: {logits.shape}")  # [batch * seq, vocab]
        print(f"flattened labels shape: {labels.shape}")  # [batch * seq]

        # 计算损失
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss



    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# ====== Step 4: Load data, vocab, tokenizer ======
# Paths (modify as needed)
vocab_path = "./enamine_real_vocabulary.txt"
data_path = "./datasetB.csv"
ckpt_output = "./llama2_finetuned.ckpt"

# Load vocab and tokenizer
vocab = load_vocabulary(vocab_path)
tokenizer = SMILESTokenizerEnamine()

# Load SMILES data
df = pd.read_csv(data_path)
smiles_list = df["Smiles"].dropna().unique().tolist()

# Dataset and dataloader
dataset = SMILESDataset(smiles_list, tokenizer, max_length=128)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=default_collate
)

# ====== Step 5: Build model from scratch ======
config = define_llama2_configuration(vocabulary_size=len(vocab), n_embd=320, n_layer=4, n_head=16)
model = Llama2FineTuner(config=config, tokenizer=tokenizer, vocab_size=len(vocab), learning_rate=5e-5)


# ====== Step 6: Start training ======
trainer = pl.Trainer(
    max_epochs=50,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath="./checkpoints",
            filename="finetuned_llama2",
            save_top_k=1,
            monitor="train_loss",
            mode="min"
        ),
        pl.callbacks.EarlyStopping(
            monitor="train_loss",
            patience=3,
            mode="min"
        )
    ],
)

trainer.fit(model, dataloader)

# Save final model checkpoint
trainer.save_checkpoint(ckpt_output)

# ====== Step 7: Create Reinforcement Learning Compatible Actor ======
from llama2 import create_llama2_actor  # Ensure this function is imported if not already

# Create actor for Reinvent framework
_, actor_inference = create_llama2_actor(vocabulary_size=len(vocab))

# Load Llama2 model weights
actor_inference.module[0].module[0].module.feature_extractor.load_state_dict(
    model.model.feature_extractor.state_dict()
)

# Load lm_head weights
actor_inference.module[0].module[1].module.load_state_dict(
    model.lm_head.state_dict()
)

# Save as reinforcement learning compatible weights
torch.save(actor_inference.state_dict(), "llama2_finetuned_A.pth")
