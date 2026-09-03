# ChekiEdgeFit-RT v2

面向拍立得照片外边框的多实例四边形提取算法。输入普通 RGB 图片，输出每个实例的检测分数、可见区域 AABB 和原图坐标下的四个外框角点。

本目录发布的是**冻结生产版本的 FP32 推理代码与最终 checkpoint**，不包含数据集、训练缓存或重新选参逻辑。版本标签为 [`cefrt2-v2.0.0`](https://github.com/czhovo/cheki/releases/tag/cefrt2-v2.0.0)。

## 算法管线

```text
原图 + EXIF 方向修正
  -> RGB uint8 / 768x768 等比例 letterbox / ImageNet 归一化
  -> RT-DETRv2-R18，300 个 query，单前景类别
  -> sigmoid(score) >= 0.9267578125，保留全部合格 query，不做 NMS
  -> 可见 AABB 还原至原图坐标
  -> square20：取长边并在四周各扩展 20%，构造正方形 ROI
  -> FP32 透视裁剪及 1024x1024 RGB resize
  -> EdgeSAM-3x：图像编码 + 原 AABB box prompt + 单 mask 解码
  -> mask logits 两阶段双线性还原 / logit > 0
  -> 冻结 G：轮廓、凸包、边线 RANSAC 拟合与相邻边求交
  -> 逆变换回原图，TL / TR / BR / BL 四角
```

检测器负责实例分离和候选数量；EdgeSAM 负责给定框对应的像素分割；G 将 mask 变成稳定的直线边框。**最终四边形不是检测器 AABB 的四个角**，也不是直接回归出的四个点。

## 模型结构

### 1. RT-DETRv2-R18 检测器

- 上游：官方 [RT-DETR](https://github.com/lyuwenyu/RT-DETR)，冻结 commit `068dfde65f2667ad6555883c69d73de886518cad`。
- Backbone：PResNet / ResNet-18-vd，输出 stride 8、16、32 的三层特征，通道数 128、256、512。
- HybridEncoder：统一为 256 通道；最高层采用一层 8-head Transformer 编码，FFN 1024；跨尺度采用卷积式特征融合。
- Decoder：RTDETRTransformerv2，3 层、hidden dimension 256、300 queries、3 个特征尺度，每层/尺度保留官方采样配置。
- 每个 query 输出一个前景 logit 和归一化 `(cx, cy, w, h)`。输入为 `[1,3,768,768]`，原始输出为 `[1,300,1]` 和 `[1,300,4]`。
- 保留训练模型内部原有的 query 选择；不额外添加推理 top-k、计数头、selector、ACR 或 NMS。

### 2. EdgeSAM-3x 分割器

- 上游：[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)，模型代码快照与 SHA-256 见 `provenance.json`。
- ImageEncoder：RepViT-M1、原始多尺度融合，输入 1024-square RGB；输出 `[1,256,64,64]` 图像嵌入。保留原始 **bicubic** 内部上采样。
- PromptEncoder：256 维坐标位置编码和 box 角点类型嵌入；使用 box prompt，无人工点、无额外 mask prompt。
- MaskDecoder：2 层 Two-Way Transformer、8 heads、embedding dimension 256、MLP dimension 2048；通过上采样特征与 mask-token 超网络生成低分辨率 logits。
- 固定 `num_multimask_outputs=1`，使用 mask token 0，输出 `[1,1,256,256]`；不从多个 mask 中另做选择。
- 分割质量分数只用于记录，**不参与筛选**。运行时保留原 Predictor 的 stability-score 语义。

| 网络 | 总参数 | 子模块参数 |
| --- | ---: | --- |
| RT-DETRv2-R18 | 20,083,028 | backbone 11,199,968；HybridEncoder 4,965,120；decoder 3,917,940 |
| EdgeSAM-3x | 9,582,112 | ImageEncoder 5,517,552；PromptEncoder 6,220；MaskDecoder 4,058,340 |

统计来自实际加载的生产模型。发布前合成张量一致性结果见 [`validation.json`](validation.json)。

### 3. 冻结几何后处理 G

`cefrt2/geometry.py` 与已评估的 `ransac_quadrilateral.py` 字节完全一致，SHA-256：

```text
92e2cd5222d115b19c3c24449e1bde98422311ebef34aca658cbe9da5cca8999
```

G 使用外轮廓、凸包、最小面积矩形初值、边线归属及固定 seed 42 的 RANSAC，默认 `filter_dist=80`、`inlier_dist=5`，最后求相邻边交点。原有内部退化处理保持不变；管线不新增“用检测框代替拟合四边形”的回退，也不加 final QNMS。拟合失败时输出 `quad=null` 和错误信息。

## 冻结细节

- 全局阈值：**0.9267578125**，由已有 cross-fitted 开发集结果冻结；本次发布不重选阈值。
- Detector letterbox 使用 OpenCV `INTER_AREA`、黑色 padding、Python ties-to-even `round`；不拉伸原图。
- Detector RGB 均值 `[0.485,0.456,0.406]`，标准差 `[0.229,0.224,0.225]`。
- square20 边长为 `max(box_width, box_height, 2.0) * 1.4`，不裁掉越界 ROI；crop side 至少 64。
- ROI 用 FP32 `warpPerspective/INTER_LINEAR`，padding RGB `[123.675,116.28,103.53]`。
- EdgeSAM RGB resize 保留 Pillow 独立 FP32 F-mode 通道 BILINEAR，不回转 uint8；标准差 `[58.395,57.12,57.375]`。
- Box prompt 坐标映射到 resize 后的 1024 图像；不能把原图坐标或归一化坐标直接传给 PromptEncoder。
- Mask 先从 256-square 双线性放大到 1024-square，再还原到 ROI 大小；均为 `align_corners=False`，之后才做 `logit > 0`。
- 神经网络与前向图像/坐标处理为 FP32，autocast 和 TF32 关闭。冻结 G 及逆变换仍包含原有 FP64 几何运算，不能擅自降精度。

## Checkpoint 下载

原始 checkpoint 作为 [GitHub Release 附件](https://github.com/czhovo/cheki/releases/tag/cefrt2-v2.0.0) 发布，不塞入 Git 历史。附件字节与本机冻结版本一致，包含原有 optimizer/RNG 等训练状态；推理只读取 `model`。请只加载可信且通过 SHA-256 验证的 PyTorch checkpoint。

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `cefrt2-rtdetr-r18-epoch24-fp32.pt` | 生产检测器，epoch24 | `65c90302a1064742b0befb67478ab58088b4973272380f9d4fbd43099b2df9c0` |
| `cefrt2-edgesam3x-epoch10-fp32.pt` | 生产分割器，固定 epoch10 | `841c35089c4b49cf4beb2b3cd09df62af60c09378a490c66bcda05bb11f4d221` |

这两个完整 `state_dict` 已覆盖推理所需的所有参数；严格加载后不需要再下载 COCO RT-DETR 或基础 EdgeSAM 权重。epoch15、交叉验证模型和中间训练快照不是本次推理资产。

## 快速运行

使用 Python 3.11，先安装与你的 CPU/CUDA 环境匹配的 PyTorch 和 torchvision，再在本目录运行：

```sh
python -m pip install -r requirements.txt
python download_checkpoints.py
python infer.py /path/to/images --output predictions.jsonl --device cpu
```

具备兼容 CUDA 环境时可显式使用 `--device cuda`。不支持 FP16、INT8、Core ML 或 MPS 推理模式切换。`tested_environment.json` 记录了本机验证版本，不代表所有依赖版本、设备后端具有逐位相同结果。

Python 调用：

```python
from cefrt2.runtime import ChekiEdgeFitRT

model = ChekiEdgeFitRT(device="cpu")
result = model.predict("/path/to/photo.jpg")
```

输出 `predictions` 中每一项含 `query_id`、`score`、`box_xyxy`、`quad`、`error`。四角顺序是 TL、TR、BR、BL，单位为 EXIF 方向修正后的原图像素，保留六位小数。零候选图片返回空数组；失败项不伪造四角。CLI 不覆盖已有输出文件。

运行测试：

```sh
python -m unittest discover -s tests -v
```

未下载权重时仅执行几何测试，模型测试会明确跳过。下载后测试会执行严格权重加载、FP32 前向和合成 box-prompt 路径，不读取任何真实图片。运行时沿用原冻结环境对未使用的 MMDetection 训练模块的兼容导入占位，不包含新增 RPN；推荐在独立进程中使用本包。

## 训练来源与已有结果

生产 RT-DETR 在开发集 4,229 图、5,371 GT 上 refit 24 epochs，固定 seed 500399。生产 EdgeSAM 使用 5,354 条严格 cross-fitted 检测框 prompts，seed 500499，训练 15 epochs，预先固定使用 **epoch10**。生产训练集上的检测框不会反向替代 cross-fitted prompts。

下表是发布前已冻结的 Python 管线结果；本次上传没有重新跑数据集或调整模型。

| 评估范围 | 图 / GT | Matched | Recall@QIoU0.5 | Precision | Strict / StrictRecall |
| --- | --- | ---: | ---: | ---: | ---: |
| Development 三折 crossfit | 4229 / 5371 | 5344 | 99.4973% | 99.7946% | 4485 / 83.5040% |
| Production 训练集 sanity | 4229 / 5371 | 5369 | 99.9628% | 99.9628% | 4889 / 91.0259% |
| 旧 consumed-test 描述性复测 | 771 / 950 | 946 | 99.5789% | 99.8944% | 816 / 85.8947% |

Strict 要求匹配四边形 QIoU >= 0.99，且角点对齐后的最大像素误差除以 GT 平均对角线 <= 0.005。检测召回高不代表所有角点都达到 Strict。

**Production sanity 是训练集检查，不是泛化成绩；旧测试集曾被使用，不能当作新的独立最终验证。** Development crossfit 是历史多折方案的证据，与此处单一生产模型不是同一次评估。发布前新增的仅为合成张量与软件封装一致性检查，结果见 `validation.json`。

此前另有 Core ML FP32 导出，但 Mac/iPhone 真机数值、速度与内存验收尚未完成；不能据此声明已经通过 iOS 部署验证。本 Release 交付 Python 算法和两个原始最终 checkpoint。

## 目录与许可证

- `cefrt2/runtime.py`：独立推理入口。
- `cefrt2/host_ops.py`：从冻结代码逐函数原样提取的预处理与坐标运算。
- `cefrt2/geometry.py`：原样冻结 G。
- `third_party/`：所需模型源码和配置，不需要完整实验仓库。
- `checkpoints.json`：固定下载地址、大小和 SHA-256。
- `provenance.json`：上游代码及原始函数来源、哈希。

第三方许可证见 [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md)。特别注意：EdgeSAM 随附 **S-Lab License 1.0**，其许可文本限定非商业用途，并要求商业使用联系贡献者；公开下载不等于额外获得商业授权。
