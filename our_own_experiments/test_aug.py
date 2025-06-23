import os
import yaml
import subprocess
import copy

def run_augmentation_experiments():
    with open("configs/base.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    instance_data_root = "data/instance"
    subjects = [s for s in os.listdir(instance_data_root) if os.path.isdir(os.path.join(instance_data_root, s))]
    print(f"Found subjects: {subjects}")

    experiments = [
        {"use_augmentation": True, "suffix": "aug_on"},
        {"use_augmentation": False, "suffix": "aug_off"},
    ]

    for subject_name in subjects:
        for exp_setting in experiments:
            print("\n" + "="*50)
            print(f"Starting experiment for Subject: '{subject_name}' with Augmentation: {exp_setting['use_augmentation']}")
            print("="*50)

            config = copy.deepcopy(base_config)
            
            exp_name = f"{subject_name}_{exp_setting['suffix']}"
            config['experiment_name'] = exp_name
            
            config['use_augmentation'] = exp_setting['use_augmentation']'
            class_name = ''.join([i for i in subject_name if not i.isdigit()]) 
            
            config['subjects'][0]['name'] = subject_name
            config['subjects'][0]['instance_data_dir'] = f"data/instance/{subject_name}"
            config['subjects'][0]['class_data_dir'] = f"data/class/{class_name}"
            config['subjects'][0]['instance_prompt'] = f"a photo of sks {subject_name}"
            config['subjects'][0]['class_prompt'] = f"a photo of a {class_name}"
            config['validation_prompt'] = f"a photo of sks {subject_name}"

            temp_config_path = f"temp_config_{exp_name}.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)

            try:
                command = ["python", "train.py", "--config_base", temp_config_path]
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during training for {exp_name}: {e}")
            finally:
                os.remove(temp_config_path)
                print(f"Finished experiment: {exp_name}")

if __name__ == "__main__":
    run_augmentation_experiments()