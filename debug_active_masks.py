import torch
import numpy as np

# 创建示例数据
batch_size = 4
hidden_size = 1

# 模拟surr1和surr2，形状为[batch_size, 1]
surr1 = torch.tensor([[0.5], [0.8], [0.3], [0.9]], dtype=torch.float32)
surr2 = torch.tensor([[0.4], [0.9], [0.2], [0.7]], dtype=torch.float32)

# 模拟active_masks_batch，形状为[batch_size, 1]
# 假设智能体1和3活跃，智能体2和4终止
active_masks_batch = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

print("=== 原始数据 ===")
print(f"surr1: {surr1}")
print(f"surr2: {surr2}")
print(f"active_masks_batch: {active_masks_batch}")
print(f"active_masks_batch.sum(): {active_masks_batch.sum().item()}")

# 步骤1: 取surr1和surr2的最小值
min_values = torch.min(surr1, surr2)
print(f"\n步骤1 - 取最小值: {min_values}")

# 步骤2: 对每个样本取负号（因为是最大化问题）
neg_values = -min_values
print(f"步骤2 - 取负号: {neg_values}")

# 步骤3: 乘以活跃掩码，过滤掉已终止智能体的损失
masked_values = neg_values * active_masks_batch
print(f"步骤3 - 乘以活跃掩码: {masked_values}")

# 步骤4: 求和，得到总损失
total_loss = masked_values.sum()
print(f"步骤4 - 求和: {total_loss.item()}")

# 步骤5: 除以活跃智能体数量，得到平均损失
avg_loss = total_loss / active_masks_batch.sum()
print(f"步骤5 - 除以活跃数量: {avg_loss.item()}")

# 完整计算
policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
print(f"\n完整计算结果: {policy_action_loss.item()}")

# 验证：手动计算
manual_loss = ((-0.4) * 1 + (-0.7) * 0 + (-0.2) * 1 + (-0.7) * 0) / 2
print(f"手动计算结果: {manual_loss}")