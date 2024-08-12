import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from datetime import datetime
import os
import re
import csv

def get_next_xid(path):
    max_id = -1
    pattern = re.compile(r'^(\d+)_')
    for filename in os.listdir(path):
        m = pattern.match(filename)
        if m:
            current_id = int(m.group(1))
            if(current_id > max_id):
                max_id = current_id
    return max_id + 1

def get_current_time_str():
    now = datetime.now()
    formatted_time = now.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time

def visualize_noisy_sample(pilot, loader, file_path=None):
    # Unzip sample_batch to 10 samples
    x, y = next(iter(loader)) # [n, 64, 1, 28, 28] -> [64, 1, 28, 28]
    
    samples = [(x[i], y[i]) for i in range(10)] # [64, 1, 28, 28] -> 10 * [1, 28, 28]
        
    # Draw 2 x 5 grid image
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i][0].permute(1,2,0), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label:{samples[i][1].item()}')

    if pilot is True:
        plt.show()
    else:
        # Output the image to path
        plt.tight_layout()
        plt.savefig(file_path)

def visualize_epoch_loss(pilot, epoch_loss, file_path=None):
    plt.figure(figsize=(10,6))
    plt.plot(epoch_loss)
    plt.title('Epoch Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    if pilot is True:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(file_path)

def visualize_confusion_matrix(pilot, all_labels, all_predictions, num_label, noise_type, accuracy, file_path=None):
    cm = confusion_matrix(all_labels, all_predictions)
    labels = list(range(num_label))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title(f'N. Type: {noise_type} | Accuracy: {accuracy}%')
    if pilot is True:
        plt.show()
    else:
        plt.savefig(file_path)

def calculate_confusion_metrics(labels, predictions, num_class=2):
    TP = TN = FP = FN = 0

    if num_class == 2:
        for label, prediction in zip(labels, predictions):
            if label == 1 and prediction == 1:
                TP += 1
            elif label == 0 and prediction == 0:
                TN += 1
            elif label == 0 and prediction == 1:
                FP += 1
            elif label == 1 and prediction == 0:
                FN += 1

        # Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # F1-Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        return [precision], [recall], [f1_score]
    
    else:
        precisions = []
        recalls = []
        f1_scores = []

        report = classification_report(labels, predictions, target_names=list(range(num_class)), output_dict=True)

        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])

        return precisions, recalls, f1_scores

def save_record_to_csv(file_path, record):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = record.keys()
            writer.writerow(header)

        writer.writerow(record.values())