import os
import joblib

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

random_state = 420

dataset = np.load('../data/database_1_fft.npz')
spec_amp = dataset['fourier']
y = dataset['label']

N = len(spec_amp[0])
fs = 500

freqs = np.fft.fftfreq(N, 1 / fs)[: N // 2]

mask_100hz = freqs < 100
X = spec_amp[:, : N // 2]
X = X[:, mask_100hz]
print(f'X.shape: {X.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=420)

svm_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=random_state)),
    ('pca', PCA(n_components=80,
                random_state=random_state)),
    ('classifier', SVC(gamma=0.45,
                       C=3.26,
                       kernel='rbf',
                       verbose=2,
                       class_weight='balanced',
                       random_state=random_state)),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

svm_pipeline.fit(X_train, y_train)
cv_score = cross_val_score(svm_pipeline,
                           X_train, y_train,
                           cv=skf,
                           n_jobs=-1,
                           scoring='balanced_accuracy')
y_pred = svm_pipeline.predict(X_test)

print(f'Balanced accuracy score: {cv_score.min() : .2f} - {cv_score.max() : .2f} ({cv_score.mean() : .2f})')
print(classification_report(y_test, y_pred))

print('Saving model to file')
joblib.dump(svm_pipeline, '../models/SVC_param_search_result.pkl')
