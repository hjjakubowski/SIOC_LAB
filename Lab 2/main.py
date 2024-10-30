import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def sample_and_hold(function, n_predictions):
    """
    Funkcja typu sample and hold, próbkowanie sygnału wejściowego
    określoną liczbę razy na przedziale.

    Args:
        function (np.ndarray): Sygnał wejściowy (ciąg wartości).
        n_predictions (int): Liczba próbek do wzięcia z sygnału.

    Returns:
        np.ndarray: Wyjściowy sygnał sample and hold.
    """
    # Wybór równomiernych próbek z input_signal
    idx = np.linspace(0, len(function) - 1, n_predictions).astype(int)
    new_y = np.zeros(n_predictions)
    
    for i, index in enumerate(idx):
        new_y[i] = function[index]
    
    # Uzupełnienie pozostałych wartości poprzednimi próbkami
    for i in range(1, len(new_y)):
        new_y[i] = new_y[i - 1] if new_y[i] == 0 else new_y[i]  # Utrzymuj wartość poprzednią

    return new_y

# Definicje funkcji
def simple_sin(x):
    return np.sin(x)

def inverted_sin(x):
    return np.where(x != 0, np.sin(1 / x), 0)  # Obsługuje przypadek x=0

def signum(x):
    return np.sign(np.sin(8 * x))

# Ustawienia
n_samples = 100
n_predictions = 200  # Zmienna dla przewidywań

# Tworzenie wartości x
x = np.linspace(-np.pi, np.pi, n_samples)

# Obliczenia sygnałów
y_simple_sin = simple_sin(x)
y_inverted_sin = inverted_sin(x)
y_signum = signum(x)

# Interpolacja
x_n2_interp = np.linspace(-np.pi, np.pi, n_predictions)
y_simple_sin_interp = sample_and_hold(y_simple_sin, n_predictions)
y_inverted_sin_interp = sample_and_hold(y_inverted_sin, n_predictions)
y_signum_interp = sample_and_hold(y_signum, n_predictions)

# Prawdziwe wartości dla interpolacji
y_simple_sin_true = simple_sin(x_n2_interp)
y_inverted_sin_true = inverted_sin(x_n2_interp)
y_signum_true = signum(x_n2_interp)

# Wizualizacja wyników
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

fig.suptitle("Interpolation of Functions Using Nearest Neighbour interpolation (Range: -π to π)")

# Wykres dla Sin(x)
axs[0].plot(x_n2_interp, y_simple_sin_true, label='Original - Sin(x)', color='green')
axs[0].plot(x_n2_interp, y_simple_sin_interp, '--', label='Interpolated', color='red')
axs[0].scatter(x_n2_interp[::5], y_simple_sin_interp[::5], color='blue', label='Interpolation Points', s=20)
axs[0].legend()

# Wykres dla Sin(1/x)
axs[1].plot(x_n2_interp, y_inverted_sin_true, label='Original - Sin(x^-1)', color='green')
axs[1].plot(x_n2_interp, y_inverted_sin_interp, '--', label='Interpolated', color='red')
axs[1].scatter(x_n2_interp[::5], y_inverted_sin_interp[::5], color='blue', label='Interpolation Points', s=20)
axs[1].legend()

# Wykres dla Sign(Sin(8x))
axs[2].plot(x_n2_interp, y_signum_true, label='Original - Signum(sin(8x))', color='green')
axs[2].plot(x_n2_interp, y_signum_interp, '--', label='Interpolated', color='red')
axs[2].scatter(x_n2_interp[::5], y_signum_interp[::5], color='blue', label='Interpolation Points', s=20)
axs[2].legend()

# Obliczenie MSE
print(f"MSE for Function Sin(x): {metrics.mean_squared_error(y_true=y_simple_sin_true, y_pred=y_simple_sin_interp):.10f}")
print(f"MSE for Function Sin(x^-1): {metrics.mean_squared_error(y_true=y_inverted_sin_true, y_pred=y_inverted_sin_interp):.10f}")
print(f"MSE for Function Sign(Sin(8x)): {metrics.mean_squared_error(y_true=y_signum_true, y_pred=y_signum_interp):.10f}")

plt.show()
