import csv
import json
import nltk
import numpy as np
import sklearn as skl
from random import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model

word_feat_dict={'ORGANIZATION':0,
        'PERSON':1,
        'LOCATION':2,
        'DATE':3,
        'TIME':4,
        'MONEY':5,
        'PERCENT':6,
        'FACILITY':7,
        'GPE':8}

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def extract_text_features(sent):
    features = [0,0,0,0,0,0,0,0,0]
    tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    named_ent = nltk.ne_chunk(tagged)
    for word in named_ent:
        if type(word) is nltk.Tree and word.label() in word_feat_dict:
            features[word_feat_dict[word.label()]] += 1
    return features

def read_csv(file_name, category_dict, do_shuffle=True, with_labels=True):
    data = []
    dataset = []
    labels = []
    csv_file = open(file_name, 'r')
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        category = row[3]
        if not(category in category_dict):
            category_dict[category] = len(category_dict)
        row[3] = category_dict[category]
        #print row[2]
        obj = json.loads(row[2])
        text_feats = ''
        if type(obj) is dict and 'title' in obj and obj["title"] is not None:
            text_feats += (obj["title"] + '\n')
        if type(obj) is dict and 'body' in obj and obj["body"] is not None:
            text_feats += obj["body"]
        text_feat = extract_text_features(text_feats)
        if (row[-1] == ''):
            dataset += [text_feat + row[1:-1]]
        else:
            row.pop(0)
            dataset += [text_feat + row]
        print len(dataset)
    #dataset.pop(0)
    if (do_shuffle):
        shuffle(dataset)
    if not(with_labels):
        return dataset
    for row in dataset:
        data += [row[0:-1]]
        labels += row[-1]
    return data, labels

def prune_nonnumber(data):
    new_data = []
    for row in data:
        tmp = []
        for column in row:
            if is_number(column):
                tmp += [float(column)]
        new_data += [tmp]
    return new_data

def main():
    category_dict = {'missing':0}
    training_data, training_labels = read_csv("train.csv", category_dict)
    training_data = prune_nonnumber(training_data)
    for row in training_data:
        print 'hi'#del row[9]
    print training_data
    threshold = int(len(training_labels) * 0.8)
    validation_data = np.array(training_data[threshold:], float)
    validation_labels = np.array(training_labels[threshold:], float)
    training_data = np.array(training_data[:threshold], float)
    training_labels = np.array(training_labels[:threshold], float)
    clf = ensemble.GradientBoostingClassifier(n_estimators=1000) #ensemble.ExtraTreesClassifier()#ensemble.RandomForestClassifier() #tree.DecisionTreeClassifier()  #skl.linear_model.SGDClassifier(penalty='elasticnet') #svm.NuSVC() #GaussianNB()
    clf.fit(training_data, training_labels)
    validation_results = clf.predict(validation_data)
    fpr, tpr, thresholds = metrics.roc_curve(validation_labels, validation_results, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print auc 
    testing_data = read_csv("test.csv", category_dict, False, False)
    testing_data = prune_nonnumber(testing_data)
    for row in testing_data:
        del row[0]
    testing_results = clf.predict(testing_data)
    np.savetxt("output_fashi5_fs10.csv", testing_results)
    print metrics.accuracy_score(validation_labels, validation_results)

if __name__ == "__main__":
    main()
