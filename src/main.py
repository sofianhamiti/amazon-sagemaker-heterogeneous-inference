import sys
import time
import subprocess
from sagemaker_training import environment

# INSTANCE GROUP NAMES
PROCESSING_GROUP = "processing_group"
TRITON_GROUP = "triton_group"


def start_process(params):
    print(f"Opening process: {params}")
    subprocess.run(params)


# def shutdown_triton(triton_host):
#     for i in range(0, 10):
#         try:
#             if i > 0:
#                 print(f"Attempting to shutdown the Triton server")
#                 time.sleep(5)
#                 sys.exit(0)
#         except Exception as e:
#             print(f"Failed to shutdown triton server in {triton_host} due to: {e}")


def main():
    env = environment.Environment()

    triton_host = env.instance_groups_dict[TRITON_GROUP]["hosts"][0]

    if env.current_instance_group == PROCESSING_GROUP:
        start_process(
            ["/usr/bin/python3", "processing.py", "--triton-host", triton_host]
        )
        # shutdown_triton(triton_host)

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
