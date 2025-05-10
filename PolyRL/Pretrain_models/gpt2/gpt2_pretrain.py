import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from typing import List

from mytokenizers import SMILESTokenizerEnamine
from gpt2 import define_gpt2_configuration, GPT2
from torch.utils.data import default_collate
from torch.utils.data import default_collate
# ====== Step 1: Load Vocabulary ======
def load_vocabulary(vocab_path: str) -> List[str]:
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab

# ====== Step 2: Custom Dataset ======
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=400):
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

# ====== Step 3: LightningModule for GPT2 ======
class GPT2FineTuner(pl.LightningModule):
    def __init__(self, config, tokenizer, vocab_size, learning_rate=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.model = GPT2(config)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.index("<pad>"))

    def forward(self, input_ids):
        attention_mask = (input_ids != vocab.index("<pad>")).long()
        hidden = self.model(input_ids, attention_mask)  # shape 可能是 [seq_len, batch, hidden_size]
        if hidden.shape[0] == input_ids.shape[1]:  # 如果是 [seq_len, batch, ...]
            hidden = hidden.permute(1, 0, 2)       # 转换成 [batch, seq_len, hidden]
        logits = self.lm_head(hidden)
        return logits


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]  # [batch, seq]
        labels = batch["labels"]        # [batch, seq]

        logits = self(input_ids)        # 应该是 [batch, seq, vocab]

        # ⚠️ 打印调试信息
        print("logits shape:", logits.shape)
        print("labels shape:", labels.shape)

        # reshape 成二维张量再计算 loss
        logits = logits.view(-1, logits.size(-1))  # [batch * seq, vocab]
        labels = labels.view(-1)                   # [batch * seq]

        print("logits reshaped:", logits.shape)
        print("labels reshaped:", labels.shape)

        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# ====== Step 4: Load data, vocab, tokenizer ======
# Paths (you can modify these as needed)
vocab_path = "./enamine_real_vocabulary.txt"
data_path = "./datasetB.csv"
ckpt_output = "./gpt2A_4.51.3.ckpt"
pretrained_ckpt = "./gpt2_enamine_real.ckpt"  # your existing .ckpt

# Load vocab and tokenizer
vocab = load_vocabulary(vocab_path)
tokenizer = SMILESTokenizerEnamine()

# Load SMILES data
df = pd.read_csv(data_path)
smiles_list = df["Smiles"].dropna().unique().tolist()

# Dataset
dataset = SMILESDataset(smiles_list, tokenizer, max_length=400)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=default_collate  # ✅ 显式加上这个
)

# ====== Step 5: Build model & load pretrained weights ======
config = define_gpt2_configuration(vocabulary_size=len(vocab), n_embd=128, n_layer=24, n_head=16)
model = GPT2FineTuner(config=config, tokenizer=tokenizer, vocab_size=len(vocab), learning_rate=5e-5)

# Load pretrained state_dict
state_dict = torch.load(pretrained_ckpt, map_location="cuda")
model.model.feature_extractor.load_state_dict(state_dict, strict=False)

# ====== Step 6: Start training ======
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath="./checkpoints",
            filename="finetuned_gpt2",
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

# Save final checkpoint
trainer.save_checkpoint(ckpt_output)
from gpt2 import create_gpt2_actor  # 🟨【新增】如果你还没导入这个函数，请添加

# 🟨 构建 Reinvent 中的 actor_inference 模型
_, actor_inference = create_gpt2_actor(vocabulary_size=len(vocab))
# 加载 GPT2 权重
actor_inference.module[0].module[0].module.feature_extractor.load_state_dict(
    model.model.feature_extractor.state_dict()
)
# 加载 lm_head 权重
actor_inference.module[0].module[1].module.load_state_dict(
    model.lm_head.state_dict()
)
# 保存为强化学习兼容权重
torch.save(actor_inference.state_dict(), "gpt2A_4.51.3.pth")