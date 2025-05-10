from mytokenizers import SMILESTokenizerEnamine

tokenizer = SMILESTokenizerEnamine()
vocab_set = set(vocab)  # vocab 是已加载词表列表

missing_tokens = set()
for smi in smiles_list:  # smiles_list 是你的 SMILES 数据
    tokens = tokenizer.tokenize(smi, with_begin_and_end=True)
    for t in tokens:
        if t not in vocab_set:
            missing_tokens.add(t)

print("以下 token 出现在数据中但不在词表中：")
for token in sorted(missing_tokens):
    print(token)
