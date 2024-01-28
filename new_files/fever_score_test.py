import argparse
import json
import sys
#from fever.scorer import fever_score
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import six
import os

def check_predicted_evidence_format(instance):
    if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
        assert all(isinstance(prediction, list)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(len(prediction) == 2
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(isinstance(prediction[0], six.string_types)
                    for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"

        assert all(isinstance(prediction[1], int)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"


def is_correct_label(instance):
    return instance["label"].upper() == instance["predicted_label"].upper()


def is_strictly_correct(instance, max_evidence=None):
    #Strict evidence matching is only for NEI class
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT ENOUGH INFO" and is_correct_label(instance):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is None:
            max_evidence = len(instance["predicted_evidence"])


        for evience_group in instance["evidence"]:
            #Filter out the annotation ids. We just want the evidence page and line number
            actual_sentences = [[e[2], e[3]] for e in evience_group]
            #Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                return True

    #If the class is NEI, we don't score the evidence retrieval component
    elif instance["label"].upper() == "NOT ENOUGH INFO" and is_correct_label(instance):
        return True

    return False


def append_lst(instance, y_true, y_pred):
    if instance["label"].upper() == "SUPPORTS":
        y_true.append(0)
    elif instance["label"].upper() == "REFUTES":
        y_true.append(1)
    elif instance["label"].upper() == "NOT ENOUGH INFO":
        y_true.append(2)


    if instance["predicted_label"].upper() == "SUPPORTS":
        y_pred.append(0)
    elif instance["predicted_label"].upper() == "REFUTES":
        y_pred.append(1)
    elif instance["predicted_label"].upper() == "NOT ENOUGH INFO":
        y_pred.append(2)


def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


# Micro is not used. This code is just included to demostrate our model of macro/micro
def evidence_micro_precision(instance):
    this_precision = 0
    this_precision_hits = 0

    # We only want to score Macro F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        for prediction in instance["predicted_evidence"]:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

    return this_precision, this_precision_hits


def fever_score(y_true, y_pred, predictions,actual=None, max_evidence=5):
    correct = 0
    strict = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for idx,instance in enumerate(predictions):
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

        #If it's a blind test set, we need to copy in the values from the actual data
        if 'evidence' not in instance or 'label' not in instance:
            assert actual is not None, 'in blind evaluation mode, actual data must be provided'
            assert len(actual) == len(predictions), 'actual data and predicted data length must match'
            assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
            instance['evidence'] = actual[idx]['evidence']
            instance['label'] = actual[idx]['label']

        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        append_lst(instance, y_true, y_pred)
        if is_correct_label(instance):
            correct += 1.0

            if is_strictly_correct(instance, max_evidence):
                strict+=1.0

        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    total = len(predictions)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    cof = confusion_matrix(y_true, y_pred)
    print(cof)

    return strict_score, acc_score, pr, rec, f1




parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--actual",type=str)
parser.add_argument('--nl_coef', type=float, default=0.0, help='a hyperparameter for negative learning')
parser.add_argument('--comp', default=None, choices=['all', 'sr', 'srn', None], help='method for negative loss computation')
parser.add_argument('--imb', action='store_true', help='with/without imbalanced learning') # default False
parser.add_argument('--beta', type=float, default=0.9999, help='a hyperparameter for imbalanced learning')
parser.add_argument('--name', help='name')
parser.add_argument('--idx', type=int, default=0, help='idx')


args = parser.parse_args()

predicted_labels =[]
predicted_evidence = []
actual = []
y_pred = []
y_true = []

ids = dict()
with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        ids[json.loads(line)["id"]] = len(predicted_labels)
        predicted_labels.append(json.loads(line)["predicted_label"])
        predicted_evidence.append(0)
        actual.append(0)



with open(args.predicted_evidence,"r") as predictions_file:
    for line in predictions_file:
        evidences = list()
        if "predicted_evidence" in json.loads(line):
            for evidence in json.loads(line)["predicted_evidence"]:
                evidences.append(evidence[:2])
            predicted_evidence[ids[json.loads(line)["id"]]] = evidences
        if "evidence" in json.loads(line):
            for evidence in json.loads(line)["evidence"]:
                evidences.append(evidence[:2])
            predicted_evidence[ids[json.loads(line)["id"]]] = evidences

with open(args.actual, "r") as actual_file:
    for line in actual_file:
        actual[ids[json.loads(line)["id"]]] = json.loads(line)

predictions = []
for ev,label in zip(predicted_evidence,predicted_labels):
    predictions.append({"predicted_evidence":ev,"predicted_label":label})

score,acc,precision,recall,f1 = fever_score(y_true, y_pred, predictions,actual)


if args.comp is None:
    new_dir_path_recursive = './ECE_' + str(args.nl_coef) + "_None_" + str(args.beta) + "_" + str(args.idx)
else:
    new_dir_path_recursive = './ECE_' + str(args.nl_coef) + "_" + args.comp + "_" + str(args.beta) + "_" + str(args.idx)
try:
    os.makedirs(new_dir_path_recursive)
except FileExistsError:
    pass


with open(new_dir_path_recursive + "/" + args.name + "_confusion.dat", "w") as f:
    for i, val in enumerate(y_true):
        f.write(str(y_true[i]) + '     ' + str(y_pred[i]) + '\n')

tab = PrettyTable()
tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

print(tab)
