from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import RScriptStep, PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Workspace, Model
from azureml.core.experiment import Experiment
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Dataset, Datastore
# from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath("./util")) 
from util.attach_compute import get_compute
from attach_compute import get_compute
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import PipelineData


# from env_variables import Env
from azureml.core.authentication import ServicePrincipalAuthentication

def main():
    # e = Env()
    # print(e.workspace_name)

    # svc_pr = ServicePrincipalAuthentication(
    # tenant_id=os.environ.get("TENANT_ID"),
    # service_principal_id=os.environ.get("AZURE_SP_ID"),
    # service_principal_password=os.environ.get("AZURE_SP_PASSWORD"))

    # # Get Azure machine learning workspace
    # ws = Workspace.get(
    #     name=os.environ.get("WORKSPACE_NAME"),
    #     subscription_id=os.environ.get("SUBSCRIPTION_ID"),
    #     resource_group=os.environ.get("AZURE_RESOURCE_GROUP")
    #     ,auth=svc_pr
    # )

    #ex = Experiment(ws, 'iris-pipeline')
    #ex.archive()

    ws = Workspace.from_config()

    print("get_workspace:")
    print(ws)
    # ws.write_config(path="", file_name="config.json")
    print("writing config.json.")

    # Get Azure machine learning cluster
    aml_compute = get_compute(ws, compute_name = 'cpu1',  vm_size = 'STANDARD_D1')

    # Data stores
    data_dir = "pipelines/modelout"
    def_data_store = ws.get_default_datastore()
    output_dir = PipelineData(name="scores", 
                          datastore=def_data_store, 
                          output_path_on_compute=data_dir)

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

    # model_path  = "outputs/model.pkl"
    # data_dir = "./outputs/pipelines/modelout/"
    # train_step = PythonScriptStep(
    #     name="Train",
    #     source_directory="./",
    #     script_name="train.py",
    #     compute_target=aml_compute,
    #     arguments=["--model_path", model_path,
    #      "--data_dir",data_dir],
    #     outputs=[output_dir],
    #     runconfig=run_config,
    #     allow_reuse=False,
    # )
    # print("Train Step created")



    ######### REGISTER ################
    # model_path = "trained-model/model.pkl"
    # register_step = PythonScriptStep(
    #     name="Register",
    #     source_directory="./",
    #     script_name="register.py",
    #     compute_target=aml_compute,
    #     arguments=["--model_path", model_path],
    #     inputs=[output_dir],
    #     runconfig=run_config,
    #     allow_reuse=False,
    # )
    # print("Register Step created")

    ######### DEPLOY ################

    # print("Uploading entry script")
    # score_path = "./deploy/deploy.py"
    # datastore = ws.get_default_datastore()
    # datastore.upload_files(files = [model_path], target_path = 'deploy/', overwrite = True,show_progress = True)
    # print("done!")


    deploy_step = PythonScriptStep(
        name="Deploy",
        source_directory="./deploy",
        script_name="deploy.py",
        compute_target=aml_compute,
        arguments=[],
        inputs=[],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Deploy Step created")


    #evaluate_step.run_after(train_step)
    # register_step.run_after(deploy_step)
    steps = [deploy_step]
    train_pipeline = Pipeline(workspace=ws, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()

    published_pipeline = train_pipeline.publish(
        name="aks-deployment-pipeline",
        description=""
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')

    pipeline_parameters = { 
        "model_name": "sklearn_regression_model.pkl" 
    }
    run = published_pipeline.submit(
        ws,
        "compute-instance-pipeline-experiment",
        pipeline_parameters
    )
    #run.wait_for_completion(show_output=True)

if __name__ == '__main__':
    main()