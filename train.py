from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import RScriptStep, PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core import Workspace, Run
from azureml.core.experiment import Experiment
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Dataset, Datastore
import os
import sys
from azureml.core.authentication import ServicePrincipalAuthentication
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', dest="model_path", required=True)
parser.add_argument('--data_dir', dest="data_dir", required=True)
args = parser.parse_args()
model_path  = args.model_path
data_dir = args.data_dir

# create output directory if it does not exist
os.makedirs(data_dir, exist_ok=True)


run = Run.get_context()
if (run.id.startswith('OfflineRun')):
    ws = Workspace.from_config()
else:
    print("using run workspace")
    ws = run.experiment.workspace

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print("getting dataset")
dataset = Dataset.get_by_name(ws, name='iris')

print(os.getcwd())
df = df = dataset.to_pandas_dataframe()
print(df)
print(df.info())
reg = 0.01

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

model = LogisticRegression(C=1/reg).fit(X, Y)
print(model)

accuracy = model.score(X, Y)
#if not os.path.exists('./outputs'):
#    os.makedirs('./outputs')
#if not os.path.exists('./outputs/models'):
#    os.makedirs('./outputs/models')

#print("Accuracy is {}".format(accuracy))
#data = {"accuracy": accuracy}
#with open('./outputs/accuracy.json', 'w') as outfile:
#    json.dump(data, outfile)

print("exporting model...")
f = open(model_path, 'wb')
pickle.dump(model, f)
f.close()
#print("local output")
#run.parent.upload_file(name="./outputs/model.pkl",
#                       path_or_stream='./outputs/model.pkl')

print("Registering model dataset")
datastore = ws.get_default_datastore()
datastore.upload_files(files = [model_path], target_path = 'trained-model/', overwrite = True,show_progress = True)

print("done!")

