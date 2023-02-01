import sys
import subprocess
import boto3
from sagemaker_training import environment

# INSTANCE GROUP NAMES
PROCESSING_GROUP = "processing_group"
TRITON_GROUP = "triton_group"


def start_process(params):
    print(f"Opening process: {params}")
    subprocess.run(params)


def shutdown_job(job_name):
    sm_client = boto3.client("sagemaker", region_name="eu-west-1")
    sm_client.stop_training_job(TrainingJobName=job_name)


def main():
    env = environment.Environment()

    triton_host = env.instance_groups_dict[TRITON_GROUP]["hosts"][0]
    job_name = env.job_name

    if env.current_instance_group == PROCESSING_GROUP:
        start_process(
            ["/usr/bin/python3", "processing.py", "--triton-host", triton_host]
        )
        shutdown_job(job_name)

    elif env.current_instance_group == TRITON_GROUP:
        start_process(["sh", "run_triton.sh"])

    else:
        raise Exception(f"Unknown instance group: {env.current_instance_group}")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(f"Failed due to {e}. exiting with returncode=1")
        sys.exit(1)
