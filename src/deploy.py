import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'external', 'lerobot-mujoco-tutorial')))


from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features

from mujoco_env.y_env import SimpleEnv

import time
import json
import torch
from PIL import Image
import torchvision

ROOT = os.path.join(current_dir, "..", "dataset", "record1")
PRETRINED_MODEL_DIR = os.path.join(current_dir, "..", "model", "train1")
xml_path = os.path.join(current_dir, "..", 'external', "lerobot-mujoco-tutorial/asset/example_scene_y.xml")

device = torch.device("cuda")

SEED = None

dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=ROOT)
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.image")
# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=1, temporal_ensemble_coeff = 0.9)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# We can now instantiate our policy with this config and the dataset stats.
policy = ACTPolicy.from_pretrained(PRETRINED_MODEL_DIR, config = cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

success_count = 0
total_runs = 10
episode_time = 30.0

log_evaluation = {
    "episode_time": episode_time,
    "success_results": []
}

total_elapsed_time = 0.0
for run in range(total_runs):
    print(f"\nRun [{run+1}/{total_runs}]")

    step = 0
    PnPEnv.reset(seed=SEED)
    policy.reset()
    policy.eval()
    save_image = True
    img_transform = torchvision.transforms.ToTensor()
    
    start_time = time.time()
    while PnPEnv.env.is_viewer_alive():
        PnPEnv.step_env()
        if PnPEnv.env.loop_every(HZ=30):
            # Check if the task is completed
            success = PnPEnv.check_success()
            if success:
                log_evaluation["success_results"].append(True)
                print('Success')
                # Reset the environment and action queue
                policy.reset()
                PnPEnv.reset(seed=SEED)
                step = 0
                save_image = False
                
                success_count += 1
                break
                
            # Get the current state of the environment
            state = PnPEnv.get_ee_pose()
            # Get the current image from the environment
            image, wirst_image = PnPEnv.grab_image()
            image = Image.fromarray(image)
            image = image.resize((256, 256))
            image = img_transform(image)
            wrist_image = Image.fromarray(wirst_image)
            wrist_image = wrist_image.resize((256, 256))
            wrist_image = img_transform(wrist_image)
            
            data = {
                'observation.state': torch.tensor([state]).to(device),
                'observation.image': image.unsqueeze(0).to(device),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                'task': ['Put mug cup on the plate'],
                'timestamp': torch.tensor([step/20]).to(device)
            }
            
            # Select an action
            action = policy.select_action(data)
            action = action[0].cpu().detach().numpy()
            # Take a step in the environment
            _ = PnPEnv.step(action)
            PnPEnv.render()
            step += 1
            success = PnPEnv.check_success()
            if success:
                log_evaluation["success_results"].append(True)
                print('Success')
                success_count += 1
                break
            
            elapsed_time = time.time() - start_time
            if elapsed_time > episode_time:
                print("Fail!!!")
                log_evaluation["success_results"].append(False)
                break

success_rate = success_count / total_runs * 100
print(f"\n✅ Success in {success_count} out of {total_runs} runs")
print(f"🎯 Success rate: {success_rate:.1f}%")

with open(os.path.join(PRETRINED_MODEL_DIR, "evaluation.json"), "w") as f:
    json.dump({
        "success_rate": success_rate,
        "success_count": success_count,
        "total_runs": total_runs,
        "episode_time": log_evaluation["episode_time"],
        "success_results": log_evaluation["success_results"]
    }, f, indent=2)