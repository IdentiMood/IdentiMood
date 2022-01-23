import argparse
from ast import arg
import json
import glob
from turtle import color
import matplotlib.pyplot as plt
import warnings
import numpy as np
import warnings
from sklearn import metrics as skMetrics

colors = ["b", "g", "r", "c", "m", "y", "k", "brown", "orange", "pink", "grey", "cyan", "olive", "peru", "navy", "teal", "dodgerblue"]
indexColor = 0
curveType = ["roc", "det", "fvf"]
models_list = [ 
    'VGG-Face', 'OpenFace',
    'Facenet', 'Facenet512',
    'DeepFace', 'DeepID',
	'Dlib', 'ArcFace'
]
metrics = ["cosine", "euclidean", "euclidean_l2"]

def checkArgs(args):
    models_list_lower = [x.lower() for x in models_list]
    args.metric = args.metric.lower()
    args.curve_type = args.curve_type.lower()
    if not args.metric in metrics:
        msg = f"Metric {args.metric} not supported"
        raise Exception(msg)
    if not args.curve_type in curveType:
        msg = f"Curve {args.curve_type} not supported"
        raise Exception(msg)
    if "*" in args.models_name:
        args.models_name = models_list
        return
    for x in range(len(args.models_name)):
        if not args.models_name[x].lower() in models_list_lower:
            msg = f"Model {args.models_name[x].lower()} not supported"
            raise Exception(msg)
        else:
            args.models_name[x] = models_list[models_list_lower.index(args.models_name[x].lower())]

def loadJson(path):
    f = open(path)
    json_content = json.load(f)
    f.close()
    return json_content

def getThresholds(obj):
    thresholds = list(list(list(obj["genuine_acceptances"].values())[0].values())[0])
    thresholds = [float(numeric_string) for numeric_string in thresholds]
    return thresholds

def getCurve(obj, curveType, metric, model):
    return list(obj[curveType][metric][model].values())

def getAreaLable(model, curve):
    c1, c2 = curve
    c1 = np.array(c1)
    c2 = np.array(c2)
    area = skMetrics.auc(np.array(c1), np.array(c2))
    return model + " " + str(round(area, 4))

def intersectAxes(a1, a2):
    maxInd = 0
    minInd = 0
    for axis in (a1, a2):
        axis = np.array(axis)
        if (axis.argmax() != 0):
            maxInd = axis.argmax()
        else:
            minInd = axis.argmin()
    return list(map(lambda e : e[:max(maxInd, minInd) + 1], (a1, a2)))

def rocPlotter(obj, metric, model):
    far = getCurve(obj, "false_acceptance_rate", metric, model)
    gar = getCurve(obj, "genuine_acceptance_rate", metric, model)
    plt.plot(far, gar, label = getAreaLable(model, (far, gar)))

def detPlotter(obj, metric, model):
    far = getCurve(obj, "false_acceptance_rate", metric, model)
    frr = getCurve(obj, "false_rejection_rate", metric, model)
    plt.plot(far, frr, label = getAreaLable(model, (far, frr)))
    plt.xscale("log")
    plt.yscale("log")

def farFrrPlotter(obj, metric, model):
    global indexColor
    far = getCurve(obj, "false_acceptance_rate", metric, model)
    frr = getCurve(obj, "false_rejection_rate", metric, model)
    (far, frr) = intersectAxes(far, frr)
    thresholds = getThresholds(obj)[:len(far)]
    plt.plot(thresholds, far, label = model, color = colors[indexColor])
    plt.plot(thresholds, frr, label = model, color = colors[indexColor])
    indexColor += 1

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", type=str)
parser.add_argument("-o", "--output", help="File containing the dataset's file paths")
parser.add_argument("-m", "--metric", help="cosine, euclidean, euclidean_l2", type=str, required=True)
parser.add_argument("-ct", "--curve-type", help="roc, det, fvf", type=str, required=True)
parser.add_argument("-ds", "--dataset", type=str)
parser.add_argument("-mn", "--models-name", nargs="+", help="VGG-Face, OpenFace, Facenet, Facenet512, DeepFace, DeepID, Dlib, ArcFace, *(all)", type=str, required=True)
parser.add_argument("-s", "--show-plot", help = "Whether to show the plots as they are computed", action="store_true")
parser.add_argument("-dw", "--dont-worry", help = "Suppress wornings", action="store_true")

args = parser.parse_args()

checkArgs(args)



files = list(map(lambda e : glob.glob(e), args.files))
if [] in files:
    msg = "File " + args.files[files.index([])] + " not found"
    raise Exception(msg)

files = list(set([el for lst in files for el in lst]))

json_contents = list(map(loadJson, files))

if args.dataset:
    lst = []
    fileDump = []
    for i in range(len(json_contents)):
        if ("dataset_name" in json_contents[i] and args.dataset == json_contents[i]["dataset_name"]):
            lst.append(json_contents[i])
            fileDump.append(files[i])
    files = fileDump
    json_contents = lst

    if json_contents == []:
        msg = f"Dataset {args.dataset} not found"
        raise Exception(msg)

for index in range(len(json_contents)):
    obj = json_contents[index]
    for model in args.models_name:
        try:
            if args.curve_type == "roc":
                rocPlotter(obj, args.metric, model)
            elif args.curve_type == "det":
                detPlotter(obj, args.metric, model)
            elif args.curve_type == "fvf":
                farFrrPlotter(obj, args.metric, model)
            print(files[index], model, obj["dataset_name"])
        except KeyError as e:
            if (args.dont_worry):
                continue
            msg = f'{e} key doesn\'t exist on {files[index]}'
            warnings.warn(msg)
        


plt.legend()
plt.title("Metric: " + args.metric + "\n Dataset: " + args.dataset)

if args.output:
    plt.savefig(args.output, dpi = 300)

if args.show_plot:
        plt.show()

plt.clf()
plt.close()

