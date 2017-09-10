#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'from_poi_to_this_person', 'from_this_person_to_poi']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

print len(my_dataset.keys())
my_dataset.pop('TOTAL', 0)

for item in my_dataset:

    if my_dataset[item]['from_poi_to_this_person'] == 'NaN' or my_dataset[item]['to_messages'] == 'NaN':
        my_dataset[item]['from_per'] = 0
    else:
        my_dataset[item]['from_per'] = float(my_dataset[item]['from_poi_to_this_person']) / float(my_dataset[item]['to_messages'])

    if my_dataset[item]['from_this_person_to_poi'] == 'NaN' or my_dataset[item]['from_messages'] == 'NaN':
        my_dataset[item]['to_per'] = 0
    else:
        my_dataset[item]['to_per'] = float(my_dataset[item]['from_this_person_to_poi']) / float(my_dataset[item]['from_messages'])


features_list = ['poi', 'from_per', 'to_per']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# print labels
# print features

fb = np.array(features)
lab = np.array(labels)
# new_data = np.vstack((fb, lab))

# new_feature = []
# new_labels = []
# for (f, l) in zip(features, labels):
#     if f > 10000 and f < 500000:
#         new_feature.append(f)
#         new_labels.append(l)
#     else:
#         print "outliner:"
#         print f
#
# features = new_feature
# labels = new_labels

# plt.scatter(features, labels)
# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()

for item in data:
    s = item[1]
    t = item[2]
    if item[0] == 1:
        plt.scatter(s, t, color='r')
    else:
        plt.scatter(s, t, color='g')

plt.xlabel('from_per')
plt.ylabel('to_per')
plt.show()



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
# clf = GaussianNB()

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
# clf = DecisionTreeClassifier()
# clf = LogisticRegression()
clf.fit(features_train, labels_train)
yped = clf.predict(features_test)
# print accuracy_score(yped, labels_test)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_list2 = ["poi", "from_per", "to_per", "shared_receipt_with_poi"]
data2 = featureFormat(my_dataset, features_list2)
labels2, features2 = targetFeatureSplit(data2)
from sklearn import cross_validation
features_train2, features_test2, labels_train2, labels_test2 = cross_validation.train_test_split(features2, labels2, test_size=0.5, random_state=1)

clf = GaussianNB()
clf = DecisionTreeClassifier()
clf.fit(features_train2, labels_train2)
pred = clf.predict(features_test2)
print("score: ", accuracy_score(labels_test2, pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from sklearn import cross_validation
from sklearn.metrics import recall_score
from sklearn.cross_validation import KFold
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
                                                                                             test_size=0.5,
                                                                                             random_state=1)

kf = KFold(len(labels), 3)
for train_indices, test_indices in kf:
    features_train = [features[item] for item in train_indices]
    features_test = [features[item] for item in test_indices]
    labels_train = [labels[item] for item in train_indices]
    labels_test = [labels[item] for item in test_indices]

clf = DecisionTreeClassifier(random_state=0)
clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print('score', score)

### use manual tuning parameter min_samples_split
clf2 = DecisionTreeClassifier(min_samples_split=5, random_state=0)
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
print('score', accuracy_score(labels_test, pred2))
print('recall', recall_score(labels_test, pred2))

dump_classifier_and_data(clf, my_dataset, features_list)
