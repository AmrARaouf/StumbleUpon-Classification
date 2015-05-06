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

def extract_text_features(sents):
    features = [0,0,0,0,0,0,0,0,0]
    for sent in nltk.sent_tokenize(sents):
        tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        named_ent = nltk.ne_chunk(tagged)
        for word in named_ent:
            if type(word) is nltk.Tree and word.label() in word_feat_dict:
                features[word_feat_dict[word.label()]] = 1
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
        """
        json_str = row.pop(2)
        text_feats = ''
        try:
            obj = json.loads(json_str)
            if type(obj) is dict and 'url' in obj and obj["url"] is not None:
                text_feats += (obj["url"] + '. ')
            if type(obj) is dict and 'title' in obj and obj["title"] is not None:
                text_feats += (obj["title"] + '. ')
            if type(obj) is dict and 'body' in obj and obj["body"] is not None:
                text_feats += obj["body"]
        except ValueError:
            text_feats=''
        text_feat = extract_text_features(text_feats)
        if (row[-1] == ''):
            dataset += [text_feat + row[1:-1]]
        else:
            row.pop(0)
            dataset += [text_feat + row]
        print len(dataset), ' ', len(dataset[-1])
        """
        #####################################################
        if (row[-1] == ''):
            dataset += [row[3:-1]]
        else:
            dataset += [row[3:]]
        ####################################################
    #dataset.pop(0) #add only if there is a title row
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
            #else:
                #print column, '\n'
        new_data += [tmp]
    return new_data

def main():
    category_dict = {'missing':0}
    training_data, training_labels = read_csv("train.csv", category_dict)
    """
    for row in training_data:
        del row[9]
    training_data = prune_nonnumber(training_data)
    """
    #####################################################
    training_data = prune_nonnumber(training_data)
    #for row in training_data:
        #del row[0]
    #####################################################
    threshold = int(len(training_labels) * 0.8)
    validation_data = np.array(training_data[threshold:], float)
    validation_labels = np.array(training_labels[threshold:], float)
    training_data = np.array(training_data[:threshold], float)
    training_labels = np.array(training_labels[:threshold], float)
    clf = ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=2) #ensemble.ExtraTreesClassifier()#ensemble.RandomForestClassifier() #tree.DecisionTreeClassifier()  #skl.linear_model.SGDClassifier(penalty='elasticnet') #svm.NuSVC() #GaussianNB()
    clf.fit(training_data, training_labels)
    validation_results = clf.predict(validation_data)
    fpr, tpr, thresholds = metrics.roc_curve(validation_labels, validation_results, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print auc 
    testing_data = read_csv("test.csv", category_dict, False, False)
    """
    for row in testing_data:
        del row[9]
    testing_data = prune_nonnumber(testing_data)
    """
    #####################################################
    testing_data = prune_nonnumber(testing_data)
    #for row in testing_data:
        #del row[0]
    #####################################################
    testing_results = clf.predict(testing_data)
    np.savetxt("output_fashi5_woner3.csv", testing_results)
    print metrics.accuracy_score(validation_labels, validation_results)

if __name__ == "__main__":
    main()
