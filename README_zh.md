# GraPhish（PhishScope）使用说明

本代码仓库对应论文 GraPhish: Self-Supervised Phishing Detection via
Website Relationship Graph Analysis。

## 数据布局
- 原始拆分：`data/dataset/phishscope/phishscope_part{0-4}`
- 预处理后的图数据：`data/dataset/graphish/graphish_part{0-4}`
- 训练生成的嵌入：`data/urlnode_embedding/graphish_part{0-4}.pt`
- 训练/验证/测试拆分使用论文命名：`phishscope10`、`phishscope20`、`phishscope30`。

## 环境准备
项目使用 `uv` 管理依赖（见 `uv.lock`/`pyproject.toml`）。

1) 同步基础环境  
```bash
uv sync
```

2) 安装 CUDA 12.1 的 GPU 版 PyTorch  
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
 "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"
```

3) 安装 CUDA 12.1 的 PyG 组件  
```bash
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

4) 安装 CUDA 12.1 的 DGL  
```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

## 运行流程
1) （可选）重新生成预处理结果  
   默认数据已就绪。如需从原始拆分重新构建：
   - 使用 `preprocess/process_phishscope_data_split.py` 生成 train/val/test
   - 使用 `preprocess/phishscope_mp_generate_split.py` 构建多元路径邻接矩阵
   - 使用 `preprocess/mask_graphish_edge_split.py` 掩码边并导出特征
   - 使用 `preprocess/extract_phishscope_neibor_split.py` 抽取邻居列表

2) 训练 GraPhish 并为每个分片生成嵌入  
```bash
uv run main_split.py --dataset graphish_part0 --use_cfg
uv run main_split.py --dataset graphish_part1 --use_cfg
uv run main_split.py --dataset graphish_part2 --use_cfg
uv run main_split.py --dataset graphish_part3 --use_cfg
uv run main_split.py --dataset graphish_part4 --use_cfg
```
超参默认读取 `configs.yml`，嵌入输出到 `data/urlnode_embedding/`。

3) 合并分片并评估（MLP 分类头）  
```bash
uv run evaluate.py
```
脚本会串联五个分片的嵌入、对齐索引，并分别报告 `phishscope10/20/30` 的指标；预测详情写入 `data/results/`，运行摘要同时追加到 `data/logs/evaluate.log`。

## 额外说明
- 可运行数据集在 `phishgmae/utils/params.py` 中定义（仅 graphish_part0–4）。
- 分片级最佳超参存放于 `configs.yml`，可在命令行覆盖。