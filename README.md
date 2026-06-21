# DepthDiff

DepthDiff —— 一种以**扩散式去噪目标**训练的、深度条件单细胞低测序深度表达增强模型;推理为从纯噪声出发的反向扩散采样。低深度表达 `x_low` 作为条件观测,模型在给定 `x_low` 与测序深度比 `d` 的条件下,学习将高斯噪声还原为高深度表达。单个深度条件模型在 25%/50%/75% 三个深度上共享。

本仓库为论文配套**代码**,用于复现方法与实验。每个数据集目录自成体系,脚本均无需命令行参数、随机种子在 `config.py` 中固定。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `PBMC3K/`, `PBMC68K/`, `PANCREAS/` | 三个数据集的完整流程代码与结果指标(各自 `results/*.csv`) |
| `alra_official.R`, `alra_runner.R` | ALRA 基线(R)运行脚本 |

各数据集目录内的脚本(以 `PBMC3K/` 为例,其余一致):

- **流程**:`data_processing.py` → `train_depthdiff.py` / `train_baselines.py` → `evaluate.py`(或一键 `run_all.py`)
- **模型与公共件**:`models.py`、`diffusion.py`、`config.py`、`utils.py`
- **基线**:`train_baselines.py`(MLP / cVAE / DCA-like AE)、`magic_baseline.py`、`extra_baselines.py`(kNN)、`scvi_baseline.py`、`dca_baseline.py`、`alra_baseline.py`
- **扩展实验**:`ablation.py`(消融)、`cross_dataset.py`(跨数据集泛化,仅 PBMC3K)、`citeseq_validation.py`(CITE-seq 正交验证,仅 PBMC3K)
- `requirements.txt`:依赖

> 说明:`scvi_baseline.py` / `dca_baseline.py` / `alra_baseline.py` 对应的 scVI / DCA / ALRA 经评估均为无监督方法、与稀疏重构口径不一致,代码与结果保留以供参考。

## 运行

```bash
cd PBMC3K   # 或 PBMC68K / PANCREAS
pip install -r requirements.txt
python run_all.py
```

`run_all.py` 依次执行 `data_processing.py`(下载原始数据并构建固定 UMI 低深度基准)、`train_depthdiff.py`、`train_baselines.py`、`evaluate.py`。各数据集 `README.md` 给出该数据集的基准设定、模型细节与完整指标列表。

## 运行环境

- `gcn_cdm`(conda):训练 / 评估 / DCA(CUDA PyTorch)
- `scvi_bench`(conda):官方 scVI(scvi-tools)

## 数据与权重(未纳入仓库)

为控制体积,以下**可再生成**的大文件未提交,克隆后由上述脚本本地重建:

- `*/data/raw/`:原始公开数据集(PBMC3K / PBMC68K 来自 10x Genomics,胰腺来自 GSE84133;CITE-seq 来自 GSE100866),由 `data_processing.py` 下载/处理
- `*/data/processed/*.npz`:配对的高/低深度矩阵
- `*/checkpoints/*.pt`:训练得到的模型权重
- `*/results/figure_data.npz`、`*/results/*.png`:图相关产物

仓库内保留各数据集 `results/` 下的指标表(`metrics*.csv`、`ablation.csv`、`cross_dataset.csv`、`citeseq_validation.csv`)及 `data/processed/` 下的小型汇总 CSV,便于直接查看结果。
