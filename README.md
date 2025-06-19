# AugMo

- Setup
```bash
conda create -n augmo python=3.10
conda activate augmo

git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

cd ..
cd lerobot_mujoco_tutorial
pip install -r requirements.txt

cd asset/objaverse
unzip plate_11.zip
```

- Collect Data
```bash
python src/collect_data.py
```

