from clearml import PipelineController
from mlops_example.preprocessing import (
    dataframe_preprocessing,
    lemmatize,
    text_preprocessing,
)
from mlops_example.visualisation import class_distribution

pipe = PipelineController(
    name="DataPrepare",
    project="Amazon reviews",
    version="0.0.9",
    packages=["./mlops-example"],
    docker="python:3.11.13-slim-bookworm",
    docker_args=(
        "--network clearml_bagel "
        "--env CLEARML_API_HOST=http://apiserver:8008 "
        "--env CLEARML_WEB_HOST=http://webserver:80 "
        "--env CLEARML_FILES_HOST=http://fileserver:8081 "
        "--env CLEARML_API_ACCESS_KEY=6WSKVKHI0X050S9E8GK8NZFFM03PC6 "
        "--env CLEARML_API_SECRET_KEY=5Hp8AQevvQ2fgeVrq7hJRER0JHoaLDl3s7zWWIeO85V1cl3LGZmIbtIGotKqxArvc4Q "
        "--env CLEARML_CPU_ONLY=1 "
    ),
    enable_local_imports=True,
)

def minimal_step(test_param):
    print("STEP STARTED!")
    print(f"test_param = {test_param}")
    return "ok"


pipe.add_function_step(
    name="minimal_step",
    function=minimal_step,
    function_kwargs={"test_param": "Hello!"},
    function_return=["result"],
    execution_queue="services",
)

pipe.start("services")
