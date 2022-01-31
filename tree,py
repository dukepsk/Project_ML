import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
from sklearn import tree


# Load Data
features = pd.read_csv("data.csv")
labels = pd.read_csv("target.csv")

# Create DataFrame with features
df = pd.DataFrame(labels)
df = df.astype(int)
labels = df

# Reshape Target
arr = np.array(labels)
target = arr.reshape(-1)

# Split Data
data_train, data_test, target_train, target_test = train_test_split(features, labels, random_state=0) #Data Split TrainingSet = 75% and TestSet = 25%

# Make scorer
scorer_ACC = make_scorer(accuracy_score)
scorer_PRE = make_scorer(precision_score, zero_division=0, pos_label=1)
scorer_REC = make_scorer(recall_score, zero_division=0, pos_label=1)
scorer_F1 = make_scorer(f1_score, pos_label=1)
scorer_ROCAUC = make_scorer(roc_auc_score)

scorings = {'accuracy':scorer_ACC,
            'precision':scorer_PRE,
            'recall': scorer_REC,
            'f1': scorer_F1,
            'rocauc': scorer_ROCAUC}

# Classifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)

# Training scores
df = pd.DataFrame()

sc = cross_validate(estimator=tree,
                        X=data_train,
                        y=target_train,
                        cv=2,
                        scoring=scorings,
                        n_jobs=-1)
                        
ACC = sc['test_accuracy']
PRE = sc['test_precision']
REC = sc['test_recall']
F1 = sc['test_f1']
ROCAUC = sc['test_rocauc']

data = {'clf': 'Tree (Train)',
        'ACC': f"{ACC.mean():6.3f}",
        'PRE': f"{PRE.mean():6.3f}",
        'REC': f"{REC.mean():6.3f} ",
        'F1': f"{F1.mean():6.3f}",
        'ROCAUC': f"{ROCAUC.mean():6.3f}"}

df = df.append(data, ignore_index=True)
df = df[['clf', 'ACC', 'PRE', 'REC', 'F1', 'ROCAUC']]

# Testing scores
tree.fit(data_train, target_train)
y_pred = tree.predict(data_test)
y_proba = tree.predict_proba(data_test)

ACC = accuracy_score(y_true=target_test, y_pred=y_pred)
PRE = precision_score(y_true=target_test, y_pred=y_pred, pos_label=1)
REC = recall_score(y_true=target_test, y_pred=y_pred, pos_label=1)
F1 = f1_score(y_true=target_test, y_pred=y_pred, pos_label=1)
ROCAUC = roc_auc_score(y_true=target_test, y_score=y_proba[:,1])

data = {'clf': 'Tree (Test)', 'ACC': ACC, 'PRE': PRE, 'REC': REC, 'F1': F1, 'ROCAUC': ROCAUC}
df = df.append(data, ignore_index=True)
df = df.set_index(['clf'])

display(df)
