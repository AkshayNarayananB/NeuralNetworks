# SC4001 Part B: Chest X-Ray Classification

## Environments and Dependencies
- Python 3.8+
- PyTorch (CUDA supported if available)
- Dependencies: `pip install -r requirements.txt`
- Hardware: Designed to run on CPU or GPU (auto-detects `cuda`).

## Seed
- Global seed `42` is set in `main.py` for reproducibility.

## How to Run
1. Ensure `chestxray_binary_64.npz` is placed in the `data/` folder.
2. Open your terminal in the root directory (`hdb_xray_project/`).
3. Run the complete pipeline via:
   `python main.py`
   
The script will automatically perform the sweeps, train the final models, generate all required tables in the console, and save all requested plots/Grad-CAM overlays into the `images/` directory.