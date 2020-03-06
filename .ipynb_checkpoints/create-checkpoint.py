from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import RScriptStep, PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Workspace, Model
from azureml.core.experiment import Experiment
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Dataset, Datastore
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath("./util")) 
from attach_compute import get_compute
from env_variables import Env
from azureml.core.authentication import ServicePrincipalAuthentication

def main():
    e = Env()
    print(e.workspace_name)

    svc_pr = ServicePrincipalAuthentication(
    tenant_id=os.environ.get("TENANT_ID"),
    service_principal_id=os.environ.get("AZURE_SP_ID"),
    service_principal_password=os.environ.get("AZURE_SP_PASSWORD"))

    # Get Azure machine learning workspace
    ws = Workspace.get(
        name=os.environ.get("WORKSPACE_NAME"),
        subscription_id=os.environ.get("SUBSCRIPTION_ID"),
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP")
        ,auth=svc_pr
    )

    #ex = Experiment(ws, 'iris-pipeline')
    #ex.archive()

    print("get_workspace:")
    print(ws)
    ws.write_config(path="", file_name="config.json")
    print("writing config.json.")

    # Get Azure machine learning cluster
    aml_compute = get_compute(
        ws,
        "train-cluster",
        "STANDARD_DS2_V2")
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    run_config = RunConfiguration(conda_dependencies=CondaDependencies.create(
        conda_packages=['numpy', 'pandas',
                        'scikit-learn', 'tensorflow', 'keras'],
        pip_packages=['azure', 'azureml-core',
                      'azureml-pipeline',
                      'azure-storage',
                      'azure-storage-blob',
                      'azureml-dataprep'])
    )
    run_config.environment.docker.enabled = True

    ######### TRAIN ################
    train_step = PythonScriptStep(
        name="Train",
        source_directory="models/python/iris/train",
        script_name="train.py",
        compute_target=aml_compute,
        arguments=[
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Train Step created")

    ######### EVALUATE ################
    evaluate_step = PythonScriptStep(
        name="Evaluate",
        source_directory="models/python/iris/evaluate",
        script_name="evaluate.py",
        compute_target=aml_compute,
        arguments=[
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Evaluate Step created")

    ######### REGISTER ################
    register_step = PythonScriptStep(
        name="Register",
        source_directory="models/python/iris/register",
        script_name="register.py",
        compute_target=aml_compute,
        arguments=[
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Register Step created")

    #evaluate_step.run_after(train_step)
    register_step.run_after(train_step)
    steps = [train_step, register_step]
    train_pipeline = Pipeline(workspace=ws, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()

    published_pipeline = train_pipeline.publish(
        name="iris-pipeline",
        description=""
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')

    pipeline_parameters = { 
        "model_name": "iris-pipeline-param" 
    }
    run = published_pipeline.submit(
        ws,
        "iris-pipeline-experiment",
        pipeline_parameters
    )
    #run.wait_for_completion(show_output=True)

if __name__ == '__main__':
    main()