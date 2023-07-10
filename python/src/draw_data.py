import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('resources/neural_network/training_data/train_dump_berthoven_i100_e20_balanced_no_chords_teacc888.csv')

train_accuracy = data['train_accuracy']
validation_accuracy = data['validation_accuracy']
train_loss = data['train_loss']
validation_loss = data['validation_loss']

plt.figure(figsize=(10, 6))
plt.xticks(range(len(train_accuracy)), range(len(train_accuracy)))
plt.yticks(ticks=np.arange(0, 1.1, step=0.1))
plt.grid(linestyle='--', linewidth=0.3)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.legend()
plt.show()

# Plot train and validation loss
plt.figure(figsize=(10, 6))
plt.xticks(range(len(train_loss)), range(len(train_loss)))
plt.yticks(ticks=np.arange(0, 1.4, step=0.1))
plt.grid(linestyle='--', linewidth=0.3)
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.show()
