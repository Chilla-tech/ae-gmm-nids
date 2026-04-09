# Training script for Autoencoder model in AE + GMM NIDS
# Includes custom callbacks for logging and learning rate scheduling.
# Usage: import and call train_autoencoder(model, X_train, X_val, epochs, batch_size)

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback

def _lr_schedule(epoch, lr):
    return 1e-3 if epoch < 5 else lr * 0.95

class PrintEveryNEpochs(Callback):
    def __init__(self, every=10):
        super().__init__()
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.every == 0:
            msg = f"Epoch {epoch + 1}/{self.params['epochs']}"
            if 'loss' in logs:
                msg += f" - loss: {logs['loss']:.4f}"
            if 'val_loss' in logs:
                msg += f" - val_loss: {logs['val_loss']:.4f}"
            print(msg)


def train_autoencoder(model, X_train, X_val, epochs=500, batch_size=32):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        LearningRateScheduler(_lr_schedule), 
        PrintEveryNEpochs(every=10)
    ]

    print("Starting Autoencoder training...")
    hist = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0, # Set to use custom callback for printing
        callbacks=callbacks
    )
    return model, hist