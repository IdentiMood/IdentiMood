import argparse
import json
import glob
import matplotlib.pyplot as plt
import warnings
import numpy as np
import warnings

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
    (far, gar) = intersectAxes(far, gar)
    plt.plot(far, gar)

def detPlotter(obj, metric, model):
    far = getCurve(obj, "false_acceptance_rate", metric, model)
    frr = getCurve(obj, "false_rejection_rate", metric, model)
    (far, frr) = intersectAxes(far, frr)
    plt.plot(far, frr)

def farFrrPlotter(obj, metric, model):
    far = getCurve(obj, "false_acceptance_rate", metric, model)
    frr = getCurve(obj, "false_rejection_rate", metric, model)
    (far, frr) = intersectAxes(far, frr)
    thresholds = getThresholds(obj)[:len(far)]
    plt.plot(thresholds, far)
    plt.plot(thresholds, frr)

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", type=str)
parser.add_argument("-o", "--output", help="File containing the dataset's file paths")
parser.add_argument("-m", "--metric", help="cosine, euclidean, euclidean_l2", type=str, required=True)
parser.add_argument("-ct", "--curve-type", help="roc, det, fvf", type=str, required=True)
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
        except KeyError as e:
            if (args.dont_worry):
                continue
            msg = f'{e} key doesn\'t exist on {files[index]}'
            warnings.warn(msg)

if args.output:
    plt.savefig(args.output, dpi = 300)

if args.show_plot:
        plt.show()

plt.clf()
plt.close()

