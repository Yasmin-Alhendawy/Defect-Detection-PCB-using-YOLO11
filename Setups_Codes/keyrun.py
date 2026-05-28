from roboflow import Roboflow

rf = Roboflow(api_key="Nywj7X3YPIzPIgflXCw6")
project = rf.workspace("pcb-egrla").project("dspcbsd")
version = project.version(1)
dataset = version.download("yolov11")