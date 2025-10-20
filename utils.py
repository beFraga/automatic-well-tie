import matplotlib.pyplot as plt
import numpy as np

def plot_seismic_and_wavelet(s_real, s_pred, wavelet, dt=1.0, title=None):
    """
    Plota a sísmica real e a reconstruída lado a lado,
    com a wavelet estimada ao lado direito.
    
    Parâmetros:
    ------------
    s_real : np.ndarray
        Sísmica real (1D ou 2D)
    s_pred : np.ndarray
        Sísmica reconstruída (1D ou 2D)
    wavelet : np.ndarray
        Wavelet estimada (1D)
    dt : float
        Amostragem temporal (ms ou s)
    title : str
        Título opcional
    """
    
    # Garantir formato 2D (n_traces x n_samples)
    s_real = np.atleast_2d(s_real)
    s_pred = np.atleast_2d(s_pred)
    
    n_traces, n_samples = s_real.shape

    t = np.arange(n_samples) * dt

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax_seis, ax_wav = axes

    # ---------- Sísmicas ----------
    vmax = max(np.abs(s_real).max(), np.abs(s_pred).max())
    
    im1 = ax_seis.imshow(
        np.hstack([s_real.T, s_pred.T]),
        cmap="seismic",
        aspect="auto",
        vmin=-vmax, vmax=vmax
    )
    ax_seis.axvline(n_traces - 0.5, color='k', linestyle='--', lw=1)
    ax_seis.set_title("Sísmica real (esq)  |  Reconstruída (dir)")
    ax_seis.set_xlabel("Traços")
    ax_seis.set_ylabel("Tempo (amostras)")
    
    # ---------- Wavelet ----------
    t_w = np.arange(len(wavelet)) * dt
    ax_wav.plot(wavelet, t_w, color='k')
    ax_wav.set_title("Wavelet estimada")
    ax_wav.set_xlabel("Amplitude")
    ax_wav.invert_yaxis()

    # ---------- Layout ----------
    fig.colorbar(im1, ax=ax_seis, fraction=0.025, pad=0.04, label='Amplitude')
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_seismic_comparison(s_real, s_syn, wavelet, title=None):
    """
    Plota N linhas, cada uma com:
      - sísmica real (linha preta)
      - sísmica sintética (linha vermelha tracejada)
      - wavelet correspondente ao lado direito
    
    Parâmetros:
    ------------
    s_real : np.ndarray
        Sísmica real com shape (n, x)
    s_syn : np.ndarray
        Sísmica sintética com shape (n, y)
    wavelet : np.ndarray
        Wavelets com shape (n, z)
    title : str
        Título opcional do gráfico
    """

    n = s_real.shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5 * n),
                             gridspec_kw={'width_ratios': [3, 1]})

    # Caso n == 1, axes não é 2D → forçar formato 2D
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        # ---------- Sísmicas ----------
        ax_seis = axes[i, 0]
        ax_seis.plot(np.arange(s_real.shape[1]), s_real[i], color='k', label='Sísmica real')
        ax_seis.plot(np.arange(s_syn.shape[1]),  s_syn[i],  color='r', linestyle='--', label='Sísmica sintética')
        ax_seis.set_title(f"Traço {i+1}")
        ax_seis.set_xlabel("Tempo")
        ax_seis.set_ylabel("Amplitude")
        ax_seis.legend(loc='upper right')
        ax_seis.grid(True, alpha=0.3)

        # ---------- Wavelet ----------
        ax_wav = axes[i, 1]
        ax_wav.plot(wavelet[i], np.arange(wavelet.shape[1]), color='b')
        ax_wav.set_title("Wavelet")
        ax_wav.set_xlabel("Amplitude")
        ax_wav.invert_yaxis()
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def plotar_amostras_como_curvas(s, s_syn, w):
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
    for i in range(s.shape[0]):
        # Cria uma figura e um conjunto de subplots (1 linha, 2 colunas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Gráfico de Comparação (Esquerda) ---
        # Transpomos os arrays com .T para que cada linha se torne uma curva
        ax1.plot(s[i], color='blue', alpha=0.7)
        ax1.plot(s_syn[i], color='red', linestyle='--', alpha=0.7)
        ax1.set_title(f'Traço {i}')
        ax1.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        # Adicionando uma legenda simples manualmente
        ax1.legend(['Real', 'Sintetico'], loc='upper right')


        # --- Gráfico Individual (Direita) ---
        # Transpomos o array w também
        ax2.plot(w[i])
        ax2.set_title(f'Wavelet')
        ax2.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        # Ajusta o layout para evitar sobreposição de títulos e eixos
        plt.tight_layout()

        # Exibe os gráficos
        plt.show()