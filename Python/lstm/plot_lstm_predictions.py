import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_lstm_predictions(model, Ttrain, Ytrain, Ttest, Ytest, num_rows=4):
    """
    Plot predictions of the LSTM encoder-decoder on training and test data.

    Parameters:
    model (torch.nn.Module): Trained LSTM encoder-decoder model.
    Ttrain, Ttest (np.array): Time arrays for training and test data.
    Ytrain, Ytest (np.array): Target data arrays for training and test data.
    num_rows (int): Number of rows of plots to display.

    Returns:
    None: Plots are displayed and saved to a file.
    """

    # Validate input arrays
    if not (Ttrain.ndim == Ttest.ndim == Ytrain.ndim == Ytest.ndim == 2):
        raise ValueError("Input arrays must be 2-dimensional.")

    # Prepare figure for plotting
    fig, axes = plt.subplots(num_rows, 2, figsize=(13, 15))
    fig.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1.)
    
    for i in range(num_rows):
        for j, (T, Y, title) in enumerate([(Ttrain, Ytrain, 'Train'), (Ttest, Ytest, 'Test')]):
            # Validate array dimensions
            if T.shape[0] != Y.shape[0]:
                raise ValueError("Time and data arrays must have the same length.")

            # Select data for plotting
            t = T[:, i] if T.shape[1] > i else T[:, 0]
            y_true = Y[:, i] if Y.shape[1] > i else Y[:, 0]
            
            # Generate model predictions
            y_pred = model.predict(torch.from_numpy(y_true[np.newaxis, :, np.newaxis]).type(torch.Tensor)).squeeze().numpy()

            # Plot data
            ax = axes[i, j]
            ax.plot(t, y_true, color=(0.2, 0.42, 0.72), label='Target')
            ax.plot(t, y_pred, color=(0.76, 0.01, 0.01), label='Prediction')
            ax.set_xlabel('$Time (sec)$')
            ax.set_ylabel('$a (\mu m/s^2)$')
            ax.set_title(title)
            if i == 0 and j == 1:
                ax.legend(bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('plots/predictions.png')
    plt.show()

# Example usage:
# plot_lstm_predictions(lstm_model, Ttrain, Ytrain, Ttest, Ytest, num_rows=4)

