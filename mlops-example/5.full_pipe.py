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
    name="data_prepare",
    base_task_id="1ecd1cacb1db4f40a362a67d629fe14f",
    parameter_override={
        "kwargs/dataset_name": "${pipeline.dataset_name}",
        "kwargs/dataset_project": "${pipeline.dataset_project}",
        "kwargs/dataset_version": "${pipeline.dataset_version}",
        "kwargs/random_state": "${pipeline.random_state}",
        "kwargs/test_size": "${pipeline.test_size}",
    },
    cache_executed_step=True,
    execution_queue="default",
)

pipe.add_step(
    name="fit_model",
    base_task_id="56920c92accb477fb1880a758749c90f",
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
    parents=["data_prepare"],
    cache_executed_step=True,
    execution_queue="default",
)


pipe.start_locally()
