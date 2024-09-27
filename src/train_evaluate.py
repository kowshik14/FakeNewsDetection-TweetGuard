import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

def train_model(model, X_train, y_train, X_test, y_test):
    csv_logger = CSVLogger('training_log.csv', separator=',', append=False)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[model_checkpoint, csv_logger])

    return model

def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'ROC-AUC Score: {roc_auc}')

    # Confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Classification report
    print(classification_report(y_test, y_pred))
