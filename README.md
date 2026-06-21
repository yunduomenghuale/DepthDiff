# DepthDiff 项目结构

**正式论文稿（唯一准稿）**：[`paper_output/DepthDiff_MDPI_中文.md`](paper_output/DepthDiff_MDPI_中文.md)

DepthDiff —— 一种以扩散式去噪目标训练的深度条件单细胞低测序深度表达增强模型；推理仅需单步前向预测。

## 目录

| 路径 | 内容 |
|---|---|
| `paper_output/DepthDiff_MDPI_中文.md` | **正式稿**（MDPI 风格，含 9 图 5 表 + 跨数据集与 CITE-seq 验证；主对比 = raw/kNN/cVAE/MLP/MAGIC/DepthDiff；AE 因与 cVAE 近乎重复已删；scVI/DCA/ALRA 仅作前人方法引用） |
| `paper_figures/` | 论文图（**fig1_workflow 为 AI 绘制的 4 面板框架图、手动放置、非脚本生成** / fig2_pbmc3k·68k·pancreas 三数据集性能复合图(含 MAGIC) / fig3_density_* 三数据集预测密度散点(7 方法面板含 MAGIC,各用 25/50/75% 深度) / fig5 marker 热图 / fig6_citeseq CITE-seq 破坏-恢复柱图）。fig4 logFC 散点已删 |
| `PBMC3K/`, `PBMC68K/`, `PANCREAS/` | 三个数据集的代码与结果（各自 `results/*.csv`） |
| `citeseq_data/` | CITE-seq 正交验证原始数据（GSE100866 CBMC） |
| `make_paper_figures.py` | 一键生成论文图（dpi=600）（三数据集性能复合图 / 三数据集预测密度散点 fig3_density_* / fig5 marker 热图 / fig6_citeseq）。`fig1_workflow`、`fig4_logfc_scatter` 函数保留但已不在 main() 调用（fig1 用 AI 图手动放置；重跑脚本不会覆盖 fig1_workflow.png）|
| `paper_tables.py` / `paper_stats.py` | 生成核心指标总表 / 统计显著性表 |
| `archive/` | 已归档的旧论文版本（不再维护） |

## 主要脚本（位于各数据集文件夹）

- 流程：`data_processing.py` → `train_depthdiff.py` / `train_baselines.py` → `evaluate.py`（或 `run_all.py`）
- 扩展实验：`magic_baseline.py`（MAGIC，主对比基线）、`extra_baselines.py`（kNN）、`ablation.py`（消融）、`figure_data.py`（图数据）、`cross_dataset.py`（跨数据集泛化）、`citeseq_validation.py`（CITE-seq 正交验证）
- 已弃用（脚本/结果仍在，但不进正式稿）：`scvi_baseline.py`、`dca_baseline.py`、`alra_baseline.py`（+ 根目录 `alra_official.R`、`alra_runner.R`）——scVI/DCA/ALRA 经评估均为无监督、与稀疏重构口径不符，已移出论文

## 运行环境

- `gcn_cdm`（conda）：训练/评估/DCA/图（CUDA torch）
- `scvi_bench`（conda）：官方 scVI（scvi-tools）

## 结果文件（每数据集 `results/`）

`metrics.csv`（raw/kNN/AE/cVAE/MLP/DepthDiff；AE 列仍在但论文已弃用）、`metrics_magic.csv`（MAGIC，主对比）、`metrics_extra.csv`(kNN)、`ablation.csv`、`cross_dataset.csv`、`citeseq_validation.csv`、`figure_data.npz`；弃用：`metrics_scvi_real.csv`/`metrics_dca_real.csv`/`metrics_alra.csv`；根目录 `paper_stats.csv`。
