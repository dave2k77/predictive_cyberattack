import matplotlib.pyplot as plt
import numpy as np
train_acc = np.array([0.83, 0.95, 0.89, 0.74, 0.72, 0.1])
val_acc = np.array([0.64, 0.63, 0.69, 0.74, 0.74, 0.1])
train_loss = np.array([0.51, 0.12, 0.57, 0.76, 0.80, 3.5])
val_loss = np.array([1.2, 2.9, 0.97, 0.78, 0.74, 3.5])
config_id = np.array([1, 2, 3, 4, 5, 6])

fig, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0].plot(train_acc, config_id, color='blue')
ax[0].set_title('Training Accuracy Trend')
ax[1].plot(val_acc, config_id, color='orange')
ax[1].set_title('Training Accuracy Trend')
ax[2].plot(train_loss, config_id, color='blue')
ax[2].set_title('Training Loss Trend')
ax[3].plot(val_loss, config_id, color='blue')
ax[3].set_title('Validation Loss Trend')
plt.show()
