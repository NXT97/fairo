import os
import mrp

if "CUDA_HOME" not in os.environ:
    raise RuntimeError(
        "Set the CUDA_HOME environment variable to compile third_party/graspnet-baseline/pointnet2 and third_party/graspnet-baseline/knn."
    )

polygrasp_setup_commands = [
    ["pip", "install", "-e", "../../../msg"],
    ["pip", "install", "-e", "../../realsense_driver"],
    ["pip", "install", "-e", "."],
]


polygrasp_shared_env = mrp.Conda.SharedEnv(
    "polygrasp",
    channels=["pytorch", "fair-robotics", "aihabitat", "conda-forge"],
    dependencies=["polymetis"],
    setup_commands=polygrasp_setup_commands,
)

mrp.process(
    name="segmentation_server",
    runtime=mrp.Conda(
        yaml="./third_party/UnseenObjectClustering/environment.yml",
        setup_commands=[
            ["pip", "install", "-e", "./third_party/UnseenObjectClustering/"]
        ]
        + polygrasp_setup_commands,
        run_command=["python", "-m", "utils.mrp_wrapper"],
    ),
)

mrp.process(
    name="grasp_server",
    runtime=mrp.Conda(
        yaml="./third_party/graspnet-baseline/environment.yml",
        setup_commands=[
            ["pip", "install", "./third_party/graspnet-baseline/pointnet2/"],
            ["pip", "install", "-e", "./third_party/graspnet-baseline/"],
        ]
        + polygrasp_setup_commands,
        run_command=["python", "-m", "graspnet_baseline.mrp_wrapper"],
    ),
)

# mrp.process(
#     name="cam_pub",
#     runtime=mrp.Conda(
#         shared_env=polygrasp_shared_env,
#         run_command=["python", "-m", "polygrasp.cam_pub_sub"],
#     ),
# )

# mrp.process(
#     name="gripper_server",
#     runtime=mrp.Conda(
#         shared_env=polygrasp_shared_env,
#         run_command=[
#             "launch_gripper.py",
#             "gripper=robotiq_2f",
#             "gripper.comport=/dev/ttyUSB0",
#         ],
#     ),
# )

mrp.process(
    name="voxel",
    runtime=mrp.Conda(
        shared_env=polygrasp_shared_env,
        run_command=["python", "-m", "voxel"],
    ),
)

mrp.process(
    name="stormgrasp",
    runtime=mrp.Conda(
        shared_env=polygrasp_shared_env,
        run_command=["python", "-m", "storm_grasp"],
    ),
)
mrp.process(
    name="empty_plate",
    runtime=mrp.Conda(
        shared_env=polygrasp_shared_env,
        run_command=["python", "-m", "empty_plate"],
    ),
)
mrp.process(
    name="grasp_object",
    runtime=mrp.Conda(
        shared_env=polygrasp_shared_env,
        run_command=["python", "-m", "grasp_object"],
    ),
)

mrp.main()
