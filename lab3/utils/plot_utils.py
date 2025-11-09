import matplotlib.pyplot as plt

def plot_histories(histories, labels, title="Porównanie dokładności walidacyjnej"):
    plt.figure(figsize=(8,5))
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_accuracy'], label=label)
    plt.title(title)
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność walidacyjna')
    plt.legend()
    plt.tight_layout()
    plt.show()
