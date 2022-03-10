import numpy as np
import matplotlib.pyplot as plt

# read test/train accuracy from disk
a = np.load("metrics/6_test_accuracy.npy")
b = np.load("metrics/6_train_accuracy.npy")
c = range(len(a))

# plot test accuracy with train accuracy
plt.plot(c, a, label="Test Accuracy")
plt.plot(c, b, label="Train Accuracy")
plt.legend()
plt.show()