import pickle
import matplotlib.pyplot as plt

with open('history.pk', 'rb') as handle:
    history = pickle.load(handle)

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
