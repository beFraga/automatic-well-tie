import matplotlib.pyplot as plt
import numpy as np
import torch


def plotar_amostras_como_curvas(real, syn, solo):
    """
    Plota cada linha dos arrays como uma curva separada.

    - s (n,x) e s_syn (n,y) são plotados juntos para comparação.
    - w (n,z) é plotado em um gráfico ao lado.
    - O eixo X de cada curva terá uma "escala unitária" (0, 1, 2, ...).

    Args:
        s (np.array): Array (n, x) onde cada uma das 'n' linhas é uma amostra.
        s_syn (np.array): Array (n, y) para comparação com 's'.
        w (np.array): Array (n, z) para o gráfico individual.
    """
    for i in range(real.shape[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(real[i], color="blue", alpha=0.7)
        ax1.plot(syn[i], color="red", linestyle="--", alpha=0.7)
        ax1.set_title(f"Plot {i}")
        ax1.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.legend(["Real", "Gerada"], loc="upper right")

        ax2.plot(solo[i])
        ax2.set_title("")
        ax2.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)

        plt.tight_layout()

        plt.show()


def adjust_data_length(data, target_length=300, device="cpu"):
    """
    Ajusta o comprimento de um dado para o tamanho fixo (300 amostras)

    Parâmetros:
    -----------
    data : torch.Tensor
        Vetor 1D contendo o dado original.
    target_length : int
        Comprimento desejado (default = 300).
    device : str
        "cpu" ou "cuda"

    Retorna:
    --------
    data_out : torch.Tensor
        Dado ajustado com exatamente `target_length` amostras.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.astype(np.float32))

    data = data.to(device).flatten()
    n = data.numel()

    # Caso 1: data maior que o alvo -> recorta centralmente
    if n > target_length:
        start = (n - target_length) // 2
        end = start + target_length
        data_out = data[start:end]

    # Caso 2: data menor que o alvo -> interpola
    elif n < target_length:
        x_old = torch.linspace(0, 1, n, device=device)
        x_new = torch.linspace(0, 1, target_length, device=device)
        data_out = torch.interp(x_new, x_old, data)

    else:
        data_out = data.clone()

    # Normaliza para [-1, 1]
    data_out = data_out / data_out.abs().max()

    return data_out


def plot(x):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(x, color="blue", alpha=0.7)
    ax.set_title("Plot")
    ax.set_xlabel("Escala Unitária (índice dentro da amostra)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_2(a, b):
    """
    Plota cada linha dos arrays como uma curva separada.

    - s (n,x) e s_syn (n,y) são plotados juntos para comparação.
    - w (n,z) é plotado em um gráfico ao lado.
    - O eixo X de cada curva terá uma "escala unitária" (0, 1, 2, ...).

    Args:
        s (np.array): Array (n, x) onde cada uma das 'n' linhas é uma amostra.
        s_syn (np.array): Array (n, y) para comparação com 's'.
        w (np.array): Array (n, z) para o gráfico individual.
    """
    for i in range(a.shape[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(a[i])
        ax1.set_title(f"Plot {i}")
        ax1.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        ax2.plot(b[i])
        ax2.set_title("")
        ax2.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)

        plt.tight_layout()

        plt.show()


def plot_2j(real, syn):
    """
    Plota cada linha dos arrays como uma curva separada.

    - s (n,x) e s_syn (n,y) são plotados juntos para comparação.
    - w (n,z) é plotado em um gráfico ao lado.
    - O eixo X de cada curva terá uma "escala unitária" (0, 1, 2, ...).

    Args:
        s (np.array): Array (n, x) onde cada uma das 'n' linhas é uma amostra.
        s_syn (np.array): Array (n, y) para comparação com 's'.
        w (np.array): Array (n, z) para o gráfico individual.
    """
    for i in range(real.shape[0]):
        plt.plot(real[i], color="blue", alpha=0.7)
        plt.plot(syn[i], color="red", linestyle="--", alpha=0.7)
        plt.suptitle(f"Plot {i}")
        plt.xlabel("Escala Unitária (índice dentro da amostra)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend(["Real", "Gerada"], loc="upper right")

        plt.show()


def plot_4(a, b, c, d):
    """
    Plota cada linha dos arrays como uma curva separada.

    - s (n,x) e s_syn (n,y) são plotados juntos para comparação.
    - w (n,z) é plotado em um gráfico ao lado.
    - O eixo X de cada curva terá uma "escala unitária" (0, 1, 2, ...).

    Args:
        s (np.array): Array (n, x) onde cada uma das 'n' linhas é uma amostra.
        s_syn (np.array): Array (n, y) para comparação com 's'.
        w (np.array): Array (n, z) para o gráfico individual.
    """
    for i in range(a.shape[0]):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 6))

        ax1.plot(a[i])
        ax1.set_title(f"Plot {i}")
        ax1.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        ax2.plot(b[i])
        ax2.set_title("")
        ax2.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)

        ax3.plot(c[i])
        ax3.set_title("")
        ax3.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax3.set_ylabel("Amplitude")
        ax3.grid(True)

        ax4.plot(d[i])
        ax4.set_title("")
        ax4.set_xlabel("Escala Unitária (índice dentro da amostra)")
        ax4.set_ylabel("Amplitude")
        ax4.grid(True)

        plt.tight_layout()

        plt.show()
