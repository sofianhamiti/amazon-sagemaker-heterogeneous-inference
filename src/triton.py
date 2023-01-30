from sagemaker_training import environment

print("======== YO! ========")

env = environment.Environment()
print(f"current_instance_type={env.current_instance_type}")
print(f"current_group_name={env.current_instance_group}")
