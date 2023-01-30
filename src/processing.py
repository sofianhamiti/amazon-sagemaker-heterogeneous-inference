import sys
import argparse
import tritonclient.grpc as grpcclient
from sagemaker_training import environment


def get_triton_server_status(triton_host):
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=f"{triton_host}:8001", verbose=True
        )
        print("Triton server status via grpc")
        triton_client.is_server_live()
        triton_client.is_server_ready()
        triton_client.is_model_ready("densenet_onnx")

    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-host", type=str)
    args, _ = parser.parse_known_args()

    # Verify instance type and instance group
    env = environment.Environment()
    print(f"current_instance_type={env.current_instance_type}")
    print(f"current_group_name={env.current_instance_group}")

    print("======== PROCESSING DATA ========")
    get_triton_server_status(args.triton_host)
