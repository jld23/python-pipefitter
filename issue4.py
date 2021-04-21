import pandas as pd
import saspy
from pipefitter.transformer import Imputer
from pipefitter.pipeline import Pipeline
from pipefitter.estimator import DecisionTree


sas = saspy.SASsession()
train = pd.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv')
train_ds = sas.df2sd(df=train, table="train_ds")

# k = {'alpha': 0.0, 'cf_level': 0.25, 'criterion': None, 'leaf_size': 5, 'max_branches': 2, 'max_depth': 6, 'n_bins': 20, 'prune': False, 'var_importance': False, 'target': 'Survived', 'nominals': ['Sex', 'Survived'], 'inputs': ['Sex', 'Age', 'Fare']}
# p = {'alpha': 0.0, 'minleafsize': 5, 'nominals': ['Sex', 'Survived'], 'input': ['Sex', 'Age', 'Fare'], 'procopts': 'maxbranch=2 maxdepth=6 intervalbins=20 assignmissing = similar ', 'prune': 'off'}
# stat = sas.sasstat()
# foo = stat.hpsplit(data=train_ds, target=k['target'], **p)
meanimp = Imputer(value=Imputer.MEAN)
modeimp = Imputer(value=Imputer.MODE)
pipe = Pipeline([meanimp, modeimp])
dtree = DecisionTree(target='Survived', inputs=['Sex', 'Age', 'Fare'], nominals=['Sex', 'Survived'])
pipe.stages.append(dtree)
pipeline_model = pipe.fit(train_ds)
pipeline_model.score(train_ds)
