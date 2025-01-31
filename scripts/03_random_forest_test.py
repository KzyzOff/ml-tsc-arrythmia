from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report
from imblearn.ensemble import BalancedRandomForestClassifier

from external_tools.utils import *

load_dotenv()
data_size = os.getenv('DATASET_SIZE')
database_path = os.getenv('DENOISED_DATASET_PATH')

X, y = load_data(database_path)

random_state = 420

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm,
    y,
    test_size=0.2,
    random_state=random_state
)

clf = BalancedRandomForestClassifier(random_state=random_state,
                                     class_weight='balanced',
                                     verbose=2,
                                     n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# y_proba = clf.predict(X_test)[:, 1]
#
# precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# average_precision = average_precision_score(y_test, y_proba)

accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy: .2f}')
print('\nClassification report:\n', classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# plt.plot(recall, precision, label=f'PR curve (av. precision = {average_precision: .2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='best')
# plt.grid()
# plt.show()
