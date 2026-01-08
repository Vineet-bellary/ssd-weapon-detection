from roboflow import Roboflow

# rf = Roboflow(api_key="apUiV7snDplgUzUnTj5Y")
# project = rf.workspace("small-object-detections-smart-surveillance-system").project("cgi-weapon-dataset-q6mia")
# version = project.version(1)
# dataset = version.download("coco")

# print("Dataset downloaded successfully...")


from roboflow import Roboflow

rf = Roboflow(api_key="apUiV7snDplgUzUnTj5Y")
project = rf.workspace("small-object-detections-smart-surveillance-system").project(
    "cgi-weapon-dataset-q6mia"
)
version = project.version(2)
dataset = version.download("coco")
