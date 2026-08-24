# Pattern 6541 Encoder — 数据协议与 Stage0 决策

日期：2026-08-21
状态：数据协议已锁定，零训练基线已完成；GPU lost，尚未启动任何微调
重要：**seen-test 与 novel-test 均未评估**

## 1. 数据协议

- 原始数据：6541 张、152 个 pattern、128 个 name。
- name：严格移除目录名末尾的 `_P<digits>`。
- source group：严格移除文件名 stem 末尾的 `_p<digits>`。
- seen 资格：`clean_count >= 50 AND unique_source_group >= 15`。
- seen：52 pattern、51 name、4046 张。
- novel：100 pattern、2494 张，训练中完全不作为正类。
- seen source-group split：train/validation/test = 65%/15%/20%。
- novel 按 name 整体划分：25 pattern novel-dev、75 pattern novel-test。
- prototype：先在 source group 内平均，再跨 source group 平均。
- 调参与选模仅使用 seen-validation 和 novel-dev。

锁定清单：`training_plans/pattern_6541_v1.json`
SHA-256 fingerprint：`8ad9899bf68f6e07b3fb587c27c8dab852219aa0c8479e98f3948ea59e02fa97`

实际划分：

| Universe | Pattern | Images |
|---|---:|---:|
| Seen train | 52 | 2613 |
| Seen validation | 52 | 611 |
| Seen test（锁定） | 52 | 822 |
| Novel dev | 25 | 587 |
| Novel test（锁定） | 75 | 1907 |

## 2. 重复与泄漏审计

- 无损坏图片。
- 1 个同 pattern 字节级/解码像素级完全重复组；保留一份。
- 无跨 pattern 完全重复。
- pHash 高置信近重复 56 对：55 对同 pattern，全部并入同一 source group；1 对跨 pattern 候选因全分彩色相似度不足而不自动合并。
- 清单验证通过：路径互斥、pattern universe 互斥、seen source group 互斥、novel name 互斥、同 pattern 近重复不跨 split。

Novel-dev few-shot 可评测范围：

| 口径 | 1-shot | 2-shot | 3-shot | 5-shot |
|---|---:|---:|---:|---:|
| All eligible pattern | 24 | 24 | 24 | 22 |
| Common set（>=10 source） | 19 | 19 | 19 | 19 |

## 3. Stage0 零训练基线

所有结果使用同一 train prototype、seen-validation、novel-dev、50 次 support 重采样；没有读取 test。

| Backbone / Feature / Region | Seen Top-1 | Seen Macro | Novel common 1-shot | Novel common 5-shot |
|---|---:|---:|---:|---:|
| ResNet18 ImageNet / avgpool / Full | 44.03% | 41.28% | 20.25% | 41.29% |
| ResNet18 ImageNet / avgpool / Bottom | 48.94% | 46.27% | 19.86% | 40.14% |
| ResNet18 ImageNet / avgpool / Fusion | 49.59% | 46.11% | 22.24% | 44.29% |
| DINOv2-S/14 / final CLS / Fusion | 32.41% | 32.19% | 12.31% | 25.62% |
| DINOv2-S/14 / final patch mean / Fusion | 42.06% | 41.47% | 17.31% | 34.94% |
| DINOv2-S/14 / last4 patch mean / Full | 49.75% | 48.23% | 21.88% | 42.74% |
| DINOv2-S/14 / last4 patch mean / Bottom | 48.77% | 47.69% | 20.89% | 41.13% |
| **DINOv2-S/14 / last4 patch mean / Fusion** | **51.88%** | **50.78%** | **22.50%** | **44.49%** |
| ConvNeXt-Tiny IN22K→IN1K / Full | 47.95% | 45.00% | 20.26% | 40.21% |
| ConvNeXt-Tiny IN22K→IN1K / Bottom | 54.17% | 52.07% | 17.82% | 38.12% |
| **ConvNeXt-Tiny IN22K→IN1K / Fusion** | **56.30%** | **54.36%** | **20.19%** | **42.39%** |

结论：

- DINO 最终 CLS 不适合该任务；必须使用末 4 层 normalized patch-token mean 的跨层平均。
- ConvNeXt-Tiny 冻结特征的 seen 闭集更强。
- DINOv2 的 novel 1/5-shot 泛化更强。
- 两者都进入 Stage1 线性探测，不依据 Stage0 单独选择主干。

## 4. 分辨率探针

DINOv2 last4 patch mean Fusion，252 相对 224：

| Resolution | Seen Top-1 | Novel 1-shot | Novel 5-shot |
|---|---:|---:|---:|
| 224 | 51.88% | 22.50% | 44.49% |
| 252 | 51.06% | 21.75% | 42.48% |

252 在三项指标均退化，因此 V1 固定 224。

## 5. 下一步

GPU 恢复后，以完全相同训练预算比较：

1. DINOv2-S/14：冻结 backbone，last4 patch mean，训练 gate + projection。
2. ConvNeXt-Tiny：冻结 backbone，训练同构 gate + projection。
3. ResNet18 保留为 sanity baseline，不作为默认最终模型。

同 name 的不同 pattern 通过 sampler 与 loss mask 双保险：同一 episode 不共现；若因以后实现变化共现，则从 SupCon 分母中完全排除。

## 6. Stage1 线性探测结果

Stage1 使用 4 组确定性训练增强特征库，backbone 完全冻结，只训练可学习 gate 与 projection；两个候选使用相同 episode、loss、LR 日程和评测随机种子。

| Backbone | Best epoch | Seen Top-1 | Seen Macro | Novel 1-shot | Novel 5-shot | Joint score |
|---|---:|---:|---:|---:|---:|---:|
| **DINOv2-S/14** | **14** | **72.18%** | **71.22%** | **33.54%** | **55.09%** | **0.5777** |
| ConvNeXt-Tiny IN22K→IN1K | 13 | 72.83% | 70.92% | 25.22% | 44.49% | 0.5289 |

联合选模规则：

`0.50 × SeenMacro + 0.25 × Novel1-shot + 0.25 × Novel5-shot`

硬门槛：相对同 backbone 的冻结基线，novel 1-shot 或 5-shot 任一下降超过 1 个百分点，不得进入 Stage2。两者均通过硬门槛，但 DINO 的联合分数明显更高，因此 Stage2 选择 DINO，计划解冻最后 2 个 Transformer block 与 final norm。

Stage1 checkpoint：

- `evaluations/pattern_6541_stage1_v1/dinov2_vits14/best_encoder.pt`
- `evaluations/pattern_6541_stage1_v1/convnext_tiny_in22k/best_encoder.pt`

两份 checkpoint 均绑定同一 manifest fingerprint，且 `test_sets_evaluated=false`。

## 7. Stage2 局部微调结果

从 DINO Stage1 最佳 checkpoint 继续，只解冻最后 2 个 Transformer block 与 final norm；重新创建 AdamW optimizer，不继承 Stage1 动量。8-way × (2 support + 2 query)，80 episode/epoch，每 2 epoch 完整验证。

| Epoch | Seen Top-1 | Seen Macro | Novel 1-shot | Novel 5-shot | Joint score |
|---:|---:|---:|---:|---:|---:|
| 0（复现Stage1） | — | 71.22% | 33.54% | 55.09% | 0.57759 |
| 2 | 72.34% | 71.50% | 34.56% | 55.57% | 0.58279 |
| 4 | 73.81% | 73.19% | 35.41% | 56.78% | 0.59643 |
| 6 | 75.78% | 74.91% | 36.20% | 57.54% | 0.60890 |
| 8 | 75.94% | 74.89% | 36.49% | 57.86% | 0.61029 |
| 10 | 77.41% | 76.63% | 36.97% | 58.44% | 0.62167 |
| **12** | **77.74%** | **76.81%** | **36.97%** | **58.69%** | **0.62318** |

第 10→12 轮联合分数只增加 0.00151，低于预注册的 0.003 延长门槛，因此不延长同一 Stage2。Stage2 最佳 checkpoint：

`evaluations/pattern_6541_stage2_v1/dinov2_vits14/best_encoder.pt`

Stage3 数值准入门槛全部满足，但是否继续解冻最后 4 个 block 尚待方案确认。seen-test 与 novel-test 仍未评估。
