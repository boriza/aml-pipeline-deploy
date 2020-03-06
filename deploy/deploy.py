from azureml.core import Run, Experiment, Workspace
import os
import sys
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
import argparse



run = Run.get_context()
if (run.id.startswith('OfflineRun')):
    print("offline run")
    
    # Get Azure machine learning workspace
    ws = Workspace.get_workspace_from_config()
else:
    print("using run workspace")
    ws = run.experiment.workspace

# datastore = ws.get_default_datastore()
# datastore.download("./", prefix="deploy", overwrite=True, show_progress=True)

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())
print(ws)
#run.register_model(model_name='iris-model',
#                    model_path="./outputs/model.pkl")


from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

environment=Environment('my-sklearn-environment')
environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'scikit-learn'
])

from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script='score.py', 
                                   source_directory='.',
                                   environment=environment)

print("Deploying model to AKS...")
# deploying the model and create a new endpoint
from azureml.core.webservice import AksEndpoint
from azureml.core.compute import ComputeTarget

#select a created compute
compute = ComputeTarget(ws, 'aks')
print("Compute defined")

namespace_name="endpointnamespace"
# define the endpoint name
endpoint_name = "myendpoint1"
# define the service name
version_name= "versiona"

model = Model(ws, 'sklearn_regression_model.pkl')
print("Model defined")

endpoint_deployment_config = AksEndpoint.deploy_configuration(tags = {'modelVersion':'firstversion', 'department':'finance'}, 
                                                              description = "my first version", namespace = namespace_name, 
                                                              version_name = version_name, traffic_percentile = 40)

endpoint = Model.deploy(ws, endpoint_name, [model], inference_config, endpoint_deployment_config, compute)
endpoint.wait_for_deployment(True)
print("Deployed")

print("Scoring")
# Scoring on endpoint
import json
test_sample = json.dumps({'data': [
    [1,2,3,4,5,6,7,8,9,10], 
    [10,9,8,7,6,5,4,3,2,1]
]})

test_sample_encoded = bytes(test_sample, encoding='utf8')
prediction = endpoint.run(input_data=test_sample_encoded)
print(prediction)
