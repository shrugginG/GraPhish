# GraPhish 

This repository contains the GraPhish code that accompanies the paper: GraPhish: Self-Supervised Phishing Detection via
Website Relationship Graph Analysis.

# Raw Dataset
Our PhishScope dataset are available at [google drive](https://drive.google.com/file/d/1Q4sF71hWTBUAuZLfcBlc5ZxtXxU2QeG-/view?usp=sharing).



## Dataset Layout
- Raw PhishScope splits: `data/dataset/phishscope/phishscope_part{0-4}`
- Processed GraPhish splits: `data/dataset/graphish/graphish_part{0-4}`
- Generated embeddings: `data/urlnode_embedding/graphish_part{0-4}.pt`
- Evaluation splits follow paper naming: `phishscope10`, `phishscope20`, `phishscope30`.

## Environment
The project uses `uv` for dependency management (see `uv.lock`/`pyproject.toml`).

1) Sync the base environment  
```bash
uv sync
```

2) Install the CUDA 12.1 GPU stack for PyTorch  
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
 "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"
```

3) Install the CUDA 12.1 PyG wheels  
```bash
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

4) Install the CUDA 12.1 DGL wheels  
```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

## Running the Pipeline
1) (Optional) Rebuild the processed splits  
   The raw splits are already prepared. If you need to regenerate processed artifacts:
   - Create train/val/test splits with `preprocess/process_phishscope_data_split.py`
   - Build meta-path adjacency matrices with `preprocess/phishscope_mp_generate_split.py`
   - Mask edges and export features with `preprocess/mask_graphish_edge_split.py`
   - Extract neighbor lists with `preprocess/extract_phishscope_neibor_split.py`

2) Train GraPhish and write embeddings for each part  
```bash
uv run main_split.py --dataset graphish_part0 --use_cfg
uv run main_split.py --dataset graphish_part1 --use_cfg
uv run main_split.py --dataset graphish_part2 --use_cfg
uv run main_split.py --dataset graphish_part3 --use_cfg
uv run main_split.py --dataset graphish_part4 --use_cfg
```
The best hyperparameters are loaded from `configs.yml`; embeddings are saved under `data/urlnode_embedding/`.

3) Evaluate the combined parts with the MLP head  
```bash
uv run evaluate.py
```
This reads all five embedding files, aligns indices across parts, and reports metrics for `phishscope10/20/30`. Prediction dumps are written to `data/results/` and the aggregated summary is appended to `data/logs/evaluate.log`.

## Configuration Notes
- All runnable datasets are defined in `phishgmae/utils/params.py` (graphish_part0–4).
- Hyperparameters per part live in `configs.yml`. Override any flag on the CLI if needed.
