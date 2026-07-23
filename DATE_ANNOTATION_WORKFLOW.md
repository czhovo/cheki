# 拍立得日期标注与余额查询

## 安装

建议使用 Python 3.11。在本目录安装最小依赖：

```powershell
python -m pip install -r .\date_annotation_requirements.txt
```

交付 ZIP 已包含可直接调用 Qwen 和余额接口的凭据。解压后必须把整个目录视为敏感文件：不要提交到 Git、发送到公开服务或展示凭据文件内容。

## 当前方法

当前流程直接处理原始拍立得图片，不需要先提取墨迹：

1. `annotate_date_qwen.py` 读取图片并按 EXIF 方向校正，转换为 RGB。
2. 图片编码为 PNG Base64 Data URL，连同 `date_polaroid_extraction_prompt.md` 发送给百炼兼容接口。
3. 使用 `qwen3.7-plus`，默认关闭思考模式，不提供日期候选、图片宽高或其他槽位。
4. 模型只识别一个手写日期，返回 `[0, 1000]` 归一化 bbox 和日期文本。
5. 本地脚本把 bbox 换算回原图像素，在原图上绘制绿色框和日期，输出 PNG。

日期文本由模型按 `YYYY.MM.DD` 或 `MM.DD` 返回；找不到可靠日期时返回 `Date: null`。

## 相关文件

- `annotate_date_qwen.py`：模型调用、结果校验、坐标换算和绘图。
- `date_polaroid_extraction_prompt.md`：原始拍立得日期识别 prompt。
- `.api_keys.json`：百炼模型 API Key，使用 `aliyun_bailian` 字段。
- `query_aliyun_savings_balance.py`：查询阿里云节省计划剩余金额。
- `.aliyun_balance_credentials.json`：余额查询专用 RAM AccessKey。
- `date_annotation_requirements.txt`：独立运行所需的 Python 依赖。

余额接口没有额外的 bearer token；认证由专用 RAM AccessKey ID/Secret 完成。两个凭据文件都应加入 `.gitignore`，不要提交或输出其中的内容。

## 处理图片

处理单张图片：

```powershell
python .\annotate_date_qwen.py .\input\1.jpg `
  -o .\outs\date_polaroid_qwen37plus `
  --thinking-budget 0
```

处理整个目录，并跳过已经存在的标注图：

```powershell
python .\annotate_date_qwen.py .\input `
  -o .\outs\date_polaroid_qwen37plus `
  --thinking-budget 0 `
  --skip-existing
```

当前脚本按图片顺序串行调用。`--limit N` 只处理排序后的前 N 张；每次请求不自动重试，避免失败后产生额外费用。

输出目录包含：

- `<编号>_annotated.png`：绘制 bbox 和日期后的原始图片。
- `raw_responses/<编号>.txt`：模型原始返回。
- `responses.json`：图片尺寸、模型配置、原始结果和像素坐标结果。

## 查询余额

模型调用结束后立即查询剩余金额：

```powershell
python .\query_aliyun_savings_balance.py
```

默认仅输出金额数值。查看节省计划实例、币种和状态：

```powershell
python .\query_aliyun_savings_balance.py --json
```

推荐执行顺序：

```powershell
python .\annotate_date_qwen.py <输入图片或目录> -o <输出目录> --thinking-budget 0
python .\query_aliyun_savings_balance.py
```
