from clearml import PipelineController

pipe = PipelineController(
    name="FullPipeline",
    project="Amazon reviews",
    version="0.0.1",
    packages=["./mlops-example"],
    docker="python:3.11.13-slim-bookworm",
    enable_local_imports=True,
    # working_dir="./mlops-example",
)

pipe.add_parameter(
    name="dataset_name",
    description="ClearML dataset name",
    default="Amazon reviews dataset",
)
pipe.add_parameter(
    name="dataset_project",
    description="ClearML project",
    default="Amazon reviews",
)
pipe.add_parameter(
    name="dataset_version",
    description="ClearML dataset version",
    default="1.2",
)
pipe.add_parameter(
    name="test_size", description="Test ratio size", default=0.2, param_type="float"
)
pipe.add_parameter(
    name="random_state", description="Random state", default=42, param_type="int"
)
pipe.add_parameter(
    name="max_features",
    description="Tf-idf features limit",
    default=1000,
    param_type="int",
)
pipe.add_parameter(
    name="analyzer",
    description="Tf-idf analyzer",
    default="word",
    param_type="str",
)

pipe.add_step(
    name="train_test_split",
    base_task_id="035490e0afab42028ceb60d47103207b",
    parameter_override={
        "KWARGS/dataset_name": "${pipeline.dataset_name}",
        "KWARGS/dataset_project": "${pipeline.dataset_project}",
        "KWARGS/dataset_version": "${pipeline.dataset_version}",
        "KWARGS/test_size": "${pipeline.test_size}",
        "KWARGS/random_state": "${pipeline.random_state}",
    },
    cache_executed_step=True,
    execution_queue="default",
)

pipe.add_step(
    name="train_processing",
    base_task_id="0063964a8b8e446bad62d0f29b22b616",
    parameter_override={
        "KWARGS/dataset_name": "${pipeline.dataset_name}",
        "KWARGS/dataset_project": "${pipeline.dataset_project}",
        "KWARGS/dataset_version": "${pipeline.dataset_version}",
    },
    cache_executed_step=True,
    parents=["train_test_split"],
    execution_queue="default",
)

pipe.add_step(
    name="test_processing",
    base_task_id="4af4992d59e148339440efa0bf696b6e",
    parameter_override={
        "KWARGS/dataset_name": "${pipeline.dataset_name}",
        "KWARGS/dataset_project": "${pipeline.dataset_project}",
        "KWARGS/dataset_version": "${pipeline.dataset_version}",
    },
    cache_executed_step=True,
    parents=["train_test_split"],
    execution_queue="default",
)

pipe.add_step(
    name="fit_model",
    base_task_id="fa99d8e3c804494e9d0ff2259f49ab77",
    parameter_override={
        "General/dataset_name": "${pipeline.dataset_name}",
        "General/dataset_project": "${pipeline.dataset_project}",
        "General/train_dataset_version": "${pipeline.dataset_version}.2",
        "General/test_dataset_version": "${pipeline.dataset_version}.3",
        "General/dataset_version": "${pipeline.dataset_version}.4",
        "General/random_state": "${pipeline.random_state}",
        "General/max_features": "${pipeline.max_features}",
        "General/analyzer": "${pipeline.analyzer}",
    },
    parents=["test_processing", "train_processing"],
    cache_executed_step=True,
    execution_queue="default",
)


pipe.start("default")
