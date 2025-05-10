import torch
from gpt2 import define_gpt2_configuration, GPT2
from mytokenizers import SMILESTokenizerEnamine
from vocabulary import Vocabulary  # 你的自定义类
from pytorch_lightning import LightningModule
from pathlib import Path

class GPT2Generator(LightningModule):
    def __init__(self, config, vocab):
        super().__init__()
        self.model = GPT2(config)
        self.lm_head = torch.nn.Linear(config.n_embd, len(vocab), bias=False)
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)
        return logits

def sample_smiles(model, vocab, max_length=128, temperature=1.0, device="cuda"):
    model.eval()
    generated = [vocab.start_token_index]
    input_ids = torch.tensor([generated], device=device)

    for _ in range(max_length):
        attention_mask = (input_ids != vocab.vocab["<pad>"]).long()
        logits = model(input_ids, attention_mask=attention_mask)
        logits = logits[:, -1, :] / temperature  # 取最后一个 token 的 logits
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == vocab.end_token_index:
            break

        generated.append(next_token)
        input_ids = torch.tensor([generated], device=device)

    smiles = vocab.decode(generated)
    return smiles

if __name__ == "__main__":
    # 路径
    vocab_path = "./enamine_real_vocabulary.txt"
    ckpt_path = "./finetuned_gpt2.ckpt"

    # 初始化分词器与词表
    tokenizer = SMILESTokenizerEnamine()
    vocab = Vocabulary.load(Path(vocab_path), tokenizer=tokenizer)

    # 构建模型 & 加载权重
    config = define_gpt2_configuration(vocabulary_size=len(vocab), n_embd=128, n_layer=24, n_head=16)
    model = GPT2Generator(config=config, vocab=vocab).to("cuda")
    state_dict = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state_dict["state_dict"])  # 只取 ckpt 中的模型部分

    # 生成 SMILES
    for i in range(10):
        smiles = sample_smiles(model, vocab, max_length=128)
        print(f"SMILES {i + 1}: {smiles}")
