from Polyscore import RF_BLR_fing  # 假设你的主函数在 RFscore.py 文件中

# 示例 SMILES 列表
test_smiles = [
    '*/C(C)=C(/*)[Si](C)(C)c1ccc([Si](C)(C)C)cc1', 
    '*/N=*/[Si](C)(C)[Si](C)(C)[Si](C)(C)[Si](*)(C)(C)[Si]C1CCCC(C)C(C)C1'        # ethanol

]

# 调用函数
print("🚀 开始预测测试 ...")
try:
    predictions = RF_BLR_fing(test_smiles)
    print("🎯 预测结果如下：")
    print(predictions)
except Exception as e:
    print("❌ 出现错误：", e)
