def plot_history(history):
    import matplotlib.pyplot as plt
    history_dict = history.history
    train_acc = history_dict['categorical_accuracy']
    train_loss = history_dict['loss']

    val_acc = history_dict['val_categorical_accuracy']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()