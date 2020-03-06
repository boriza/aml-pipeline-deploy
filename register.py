from azureml.core import Run, Experiment, Workspace
import os
import sys
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', dest="model_path", required=True)
args = parser.parse_args()
model_path  = args.model_path
    


run = Run.get_context()
if (run.id.startswith('OfflineRun')):
    print("offline run")
    
    # Get Azure machine learning workspace
    ws = Workspace.get_workspace_from_config()
else:
    print("using run workspace")
    ws = run.experiment.workspace

datastore = ws.get_default_datastore()
datastore.download("./", prefix="trained-model", overwrite=True, show_progress=True)

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())
print(ws)
#run.register_model(model_name='iris-model',
#                    model_path="./outputs/model.pkl")
print("registering model...")
model = Model.register(workspace=ws,
                    model_name='iris-model', 
                    model_path=model_path)
print("registered")
