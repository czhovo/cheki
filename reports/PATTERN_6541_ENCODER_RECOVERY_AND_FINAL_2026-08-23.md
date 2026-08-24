# Pattern 6541 编码器：记录恢复与最终报告

日期：2026-08-23
状态：训练、模型选择、相邻 checkpoint 复核与锁定 test 均已完成

## 1. 恢复结论

Codex 界面确实丢失/压缩了大量中间上下文，但工作本身没有丢失。以下来源仍完整：

- 当前 Codex 任务历史：保留了浏览器咨询、命令、编辑和阶段性进度。
- 持久目标：目标正文仍可读取；早期“blocked”状态是内置浏览器不可用时留下的过时状态，不代表当前工作状态。
- 数据清单与审计 JSON。
- Stage0/1/2/3 的逐轮 validation、training history 和 checkpoint。
- ChatGPT Pro 的 `pattern encoder` 对话。
- 在 test 前写入的最终模型选择记录。
- 唯一一次 locked-test 报告。

本报告依据文件、checkpoint 哈希和 JSON 指标重建，而不是依赖模型对旧对话的记忆。

## 2. 数据与防泄漏协议

原始数据：`pattern_6541`，6541 张、152 个 pattern、128 个 name。

规则：

- `name`：移除 pattern 目录名末尾的 `_P<digits>`。
- `source_group`：移除文件名 stem 末尾的 `_p<digits>`。
- 训练正类：`clean_count >= 50 AND unique_source_group >= 15`。
- 同 name 的不同 pattern：永不在同一 episode 作为负类；若共现，则从 SupCon 分母完全排除。
- prototype：先在 source group 内平均，再跨 source group 平均。

锁定划分：

| Universe | Pattern | Images |
|---|---:|---:|
| Seen train | 52 | 2613 |
| Seen validation | 52 | 611 |
| Seen test | 52 | 822 |
| Novel dev | 25 | 587 |
| Novel test | 75 | 1907 |

Manifest fingerprint：`8ad9899bf68f6e07b3fb587c27c8dab852219aa0c8479e98f3948ea59e02fa97`

重复审计：

- 0 张损坏图片。
- 1 个同 pattern 完全重复组；保留一份。
- 0 个跨 pattern 完全重复组。
- 55 对同 pattern 高置信近重复被绑定到同一 source group，防止跨 split。

## 3. 模型选择过程

### Stage0：冻结特征

严格使用同一数据清单、prototype 与 few-shot seeds。

| 模型/表示 | Seen Top-1 | Novel 1-shot | Novel 5-shot |
|---|---:|---:|---:|
| ResNet18 Fusion | 49.59% | 22.24% | 44.29% |
| ConvNeXt-Tiny Fusion | 56.30% | 20.19% | 42.39% |
| DINOv2 最终 CLS Fusion | 32.41% | 12.31% | 25.62% |
| DINOv2 last4 patch mean Fusion | 51.88% | 22.50% | 44.49% |

结论：DINO 最终 CLS 不适合该任务；应使用最后 4 层 normalized patch-token mean 的跨层平均。252 输入在三项指标上退化，因此固定 224。

### Stage1：冻结 backbone，训练门控和投影头

| Backbone | Seen Macro | Novel 1-shot | Novel 5-shot | Joint score |
|---|---:|---:|---:|---:|
| DINOv2-S/14 | 71.22% | 33.54% | 55.09% | 0.5777 |
| ConvNeXt-Tiny | 70.92% | 25.22% | 44.49% | 0.5289 |

DINO 对未见 pattern 泛化明显更好，因此进入 Stage2。

### Stage2：解冻最后 2 个 Transformer block

最佳邻近 checkpoint 实际是 epoch 11：joint score 0.62488；epoch 12 为 0.62318。两者均远低于后续 Stage3。

### Stage3：解冻最后 4 个 Transformer block

采用 BF16、分层学习率、8-way × (2 support + 2 query)。一次 Stage3 尝试在验证期间触发 `0x116 VIDEO_TDR_FAILURE`；系统重启后，使用 60 秒验证前冷却、batch 12 和独立后台日志完成安全重试。

| Epoch | Seen Top-1 | Seen Macro | Novel 1-shot | Novel 5-shot | Score |
|---:|---:|---:|---:|---:|---:|
| 4 | 79.05% | 78.70% | 39.02% | 60.53% | 0.64239 |
| 5 | 79.87% | 79.60% | 39.52% | 61.15% | 0.64966 |
| 6 | 80.85% | 80.85% | 40.14% | 62.18% | 0.66008 |
| 7 | 81.01% | 80.97% | 40.45% | 62.12% | 0.66127 |
| **8** | **81.18%** | **81.12%** | **40.50%** | **62.14%** | **0.66223** |

epoch 7 与 8 一致，排除了偶然峰值。最终模型在 test 前锁定为 Stage3 epoch 8。

## 4. 最终模型

- Backbone：DINOv2 ViT-S/14。
- 输入：224×224，全图 + 底部 40–50% 两区域。
- 特征：最后 4 个 Transformer block 的 normalized patch-token mean，再跨层平均。
- 融合：共享 backbone；可学习双路 softmax gate。
- 投影：384 → 384 → 256，L2 normalization。
- 损失：Prototype Loss + `0.25 ×` name-masked supervised contrastive loss。
- 训练：Stage3 解冻 blocks 8–11 和 final norm，其余冻结。

最终 checkpoint：

`evaluations/pattern_6541_stage3_v1_retry2/dinov2_vits14/best_encoder.pt`

SHA-256：`fe5bcb95f15836ab8d664398ee7acede0fb700ce3ac85ffa0d60adf356a2fb01`

该 SHA 同时匹配：

- test 前的 `selection.json`；
- checkpoint 实际文件；
- 最终 `test_report.json`。

## 5. 唯一一次 locked-test 结果

最终 checkpoint 在 test 前已锁定；test 未用于模型选择，test 后未继续调参。

### Seen test

- 822 张、52 pattern。
- Top-1：**79.68%**。
- Macro accuracy：**79.05%**。
- Macro F1：**77.57%**。

相对 validation：Top-1 下降 1.50 pp，Macro 下降 2.07 pp，泛化差距有限。

### Novel test：公共 61-pattern 集

每个 shot 使用相同 query pool，50 次随机 support source-group 抽样。

| Shot | Accuracy mean | Std | Name-masked mean |
|---:|---:|---:|---:|
| 1 | **23.97%** | 1.72% | 24.20% |
| 2 | **33.31%** | 1.58% | 33.65% |
| 3 | **38.59%** | 1.67% | 38.98% |
| 5 | **45.18%** | 1.33% | 45.63% |

61 类随机机会约 1.64%。准确率显著高于随机机会，并随支持 source group 从 1 增至 5 单调提升，证明了对训练中完全未见 pattern 的可注册、可分类能力。

### Novel test：每个 shot 的全部可评测 pattern

| Shot | Patterns | Accuracy mean | Std |
|---:|---:|---:|---:|
| 1 | 74 | 22.29% | 1.55% |
| 2 | 74 | 30.80% | 1.23% |
| 3 | 73 | 35.97% | 1.43% |
| 5 | 69 | 43.40% | 1.24% |

Novel-test 明显难于 novel-dev，说明存在真实分布差异；该差距没有用于事后调参，应作为诚实的最终泛化结果保留。

## 6. 训练过程诊断

Stage3 epoch 8：

- episode query accuracy：90.70%。
- query Top-1/Top-2 cosine margin：0.231。
- 同 pattern 正相似度：0.555。
- 不同 name 负相似度：0.111。
- gate 平均权重：Full 0.551 / Bottom 0.449。
- gradient norm（裁剪前）：19.43；训练使用 clip 1.0。

这些指标与完整 validation 同时改善，训练不是仅靠 loss 下降判断。

## 7. 主要产物

- `training_plans/pattern_6541_v1.json`
- `training_plans/pattern_6541_v1_config.json`
- `reports/pattern_6541_duplicate_audit_v1.json`
- `reports/pattern_6541_manifest_verification_v1.json`
- `evaluations/pattern_6541_stage0_v1/`
- `evaluations/pattern_6541_stage1_v1/`
- `evaluations/pattern_6541_stage2_v1/`
- `evaluations/pattern_6541_stage3_v1_retry2/`
- `evaluations/pattern_6541_neighbor_eval/stage2_epoch11_validation.json`
- `evaluations/pattern_6541_final_v1/selection.json`
- `evaluations/pattern_6541_final_v1/test_report.json`

## 8. 完成判断

原目标已经实质完成：

- 使用 `pattern_6541` 重新训练了新编码器；
- 只用高样本 pattern 作为对比学习正类；
- 同 name 的不同 pattern 未作为负类；
- 严格区分 seen、novel-dev、novel-test；
- 在 seen-test 上达到 79.68% Top-1；
- 在 61 个完全未见 pattern 上展示了从 1-shot 23.97% 到 5-shot 45.18% 的可扩展分类能力；
- 最终 test 在锁模后只运行一次，没有 test 调参。

后续若继续工作，应被视为新的优化阶段，而不是当前交付缺失。

## 9. 用户追加的 novel-test 6/8/10-shot 评测

该测试在最终模型锁定之后追加；没有重新训练或重新选择checkpoint。候选范围仍为每个shot中所有具备足够source group的novel-test pattern，未执行12候选或unknown/unassigned测试。

### All-eligible

| Shot | Pattern | Primary accuracy | Std | Name-masked accuracy |
|---:|---:|---:|---:|---:|
| 6 | 66 | **46.17%** | 1.16% | 46.63% |
| 8 | 61 | **49.71%** | 1.31% | 50.21% |
| 10 | 58 | **51.98%** | 1.08% | 52.46% |

资格条件仍为 `unique_source_group >= shot + 2`，因此shot增加时可评测pattern数量下降。

### 固定公共集

为让6/8/10-shot在完全相同的pattern与query pool上比较，公共集限定为至少15个source group的55个pattern：前10个source group构成固定support pool，其余至少5个构成固定query pool。

| Shot | Pattern | Primary accuracy | Std | Name-masked accuracy |
|---:|---:|---:|---:|---:|
| 6 | 55 | **48.42%** | 1.36% | 48.84% |
| 8 | 55 | **51.17%** | 1.20% | 51.60% |
| 10 | 55 | **52.82%** | 0.96% | 53.24% |

结果随可观察source group数量单调提升，且10-shot方差最低。完整产物：

`evaluations/pattern_6541_final_v1/novel_test_extra_shots_6_8_10.json`

## 10. 用户追加的 Known 12-candidate 测试

该测试假定待分类图片必然属于随机选择的12个候选pattern，因此强制输出12类之一；没有`unassigned`，也没有执行unknown测试。

为公平比较shot数量，每个trial的12候选、query pool和10-source support pool固定；较小shot使用该support pool的嵌套前缀。共50个随机trial。同name的P1/P2若同时被选中，仍作为不同pattern正常竞争。

### Seen-test

Seen pattern 已经拥有完整训练集，因此不使用shot口径。每个被选中的候选pattern使用全部seen-train source group建立prototype。

| Prototype | 12-way accuracy | Std | Macro accuracy |
|---|---:|---:|---:|
| 全部seen-train source groups | **90.30%** | 3.88% | **90.34%** |

### Novel-test

为保证1–10-shot使用同一候选池，只从至少12个source group的58个novel-test pattern中抽取12类。

| Shot | 12-way accuracy | Std | Macro accuracy |
|---:|---:|---:|---:|
| 1 | **45.75%** | 6.33% | 44.62% |
| 2 | **56.80%** | 5.98% | 56.00% |
| 3 | **61.63%** | 5.72% | 60.59% |
| 5 | **68.08%** | 5.11% | 67.06% |
| 6 | **69.95%** | 4.94% | 68.68% |
| 8 | **71.84%** | 5.48% | 70.31% |
| 10 | **72.72%** | 5.29% | 71.55% |

候选限制为12类后，seen使用完整prototype达到90.30%；known novel 10-shot从全候选协议的约52%提高到72.72%。完整产物：

`evaluations/pattern_6541_candidate12_known_v1/report.json`

## 11. 用户追加的 Unknown 12-candidate 单阈值测试

只允许一个全局余弦阈值，并且只能使用seen-validation实验选择。阈值目标更重视候选内正确分类：

`0.70 × 候选内正确接受率 + 0.30 × 候选外 unassigned 率`

在50次seen-validation随机12候选实验上得到唯一阈值：

`cosine threshold = 0.6184226274`

Calibration：候选内正确接受86.04%，候选外unassigned79.71%，70/30目标84.14%。阈值锁定后未使用任何test数据调整。

测试协议：

- Seen：12个seen候选，使用全部seen-train source group建立prototype。
- Novel：12个novel候选，每类10-shot。
- Mixed：6个seen＋6个novel；seen完整prototype，novel 10-shot。
- 每个trial候选内/候选外query数量相等，共50个trial。
- 候选内只有“被接受且分类正确”才算正确；候选外只有输出`unassigned`才算正确。

| Test | 候选内正确接受 | 候选外unassigned | 50/50总体准确率 | 70/30目标准确率 |
|---|---:|---:|---:|---:|
| Seen | **82.34% ± 4.63%** | **79.32% ± 4.45%** | **80.83% ± 3.22%** | **81.43% ± 3.51%** |
| Novel 10-shot | **58.56% ± 5.59%** | **71.67% ± 3.77%** | **65.12% ± 2.94%** | **62.49% ± 3.78%** |
| Mixed 6+6 | **69.32% ± 6.70%** | **75.97% ± 3.69%** | **72.65% ± 3.29%** | **71.32% ± 4.47%** |

附加诊断：

- Seen候选内接受率87.53%，其中约5.19个百分点为接受后分错。
- Novel候选内接受率70.07%，其中约11.51个百分点为接受后分错。
- Mixed候选内接受率78.38%，其中约9.06个百分点为接受后分错。

完整产物：

`evaluations/pattern_6541_candidate12_unknown_v1/report.json`
