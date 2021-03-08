# Importez les données digits avec dataset.load_digits.
# Divisez votre échantillon en deux parties. Effectuez un svm
# pour identifier les chiffres et testez sur l’échantillon de test.

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

svm = SVC().fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

svm = SVC(max_iter=10).fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(f"(max_iter=10) Accuracy: {metrics.accuracy_score(y_test, y_pred)}")