from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

def plotar_amostras_como_curvas(real, syn, solo):
    """
    Garante que o array tem exatamente 2 dimensões (n_amostras, n_features).
    Cria cópia se necessário para evitar problemas de broadcast.
    """
    data = np.atleast_1d(np.asarray(data))

    if data.ndim == 1:
        data = data.reshape(1, -1).copy()
    elif data.ndim == 2:
        data = data.copy()
    else:
        n_first = data.shape[0]
        data = data.reshape(n_first, -1).copy()

    return data


def _alinhar_tamanhos_amostras(
    real: np.ndarray, syn: np.ndarray, solo: np.ndarray
) -> tuple:
    """
    Alinha os tamanhos dos arrays de amostras.

    Problemas comuns:
    - syn pode vir de convolução com tamanhos variáveis
    - real, syn podem ter comprimentos diferentes

    Estratégia: usar o menor tamanho comum para todas as amostras.
    """
    n_samples = min(real.shape[0], syn.shape[0], solo.shape[0])

    # Pegar apenas as primeiras n_samples
    real = real[:n_samples].copy()
    syn = syn[:n_samples].copy()
    solo = solo[:n_samples].copy()

    # Encontrar o tamanho mínimo de features entre todos
    # (cada amostra pode ter tamanho diferente)
    min_feat_real_syn = min(real.shape[1], syn.shape[1])

    # Truncar para o tamanho mínimo
    real = real[:, :min_feat_real_syn].copy()
    syn = syn[:, :min_feat_real_syn].copy()

    return real, syn, solo


def _calcular_metricas(real: np.ndarray, syn: np.ndarray) -> dict:
    """Calcula RMSE e correlação entre sinais."""
    # Garantir que têm o mesmo tamanho
    min_len = min(len(real), len(syn))
    real = real[:min_len]
    syn = syn[:min_len]

    rmse = np.sqrt(np.mean((real - syn) ** 2))
    try:
        correlacao = np.corrcoef(real, syn)[0, 1]
        if np.isnan(correlacao):
            correlacao = 0.0
    except:
        correlacao = 0.0
    return {"rmse": rmse, "correlacao": correlacao}


def plotar_amostras_como_curvas(
    real: Union[np.ndarray, torch.Tensor],
    syn: Union[np.ndarray, torch.Tensor],
    solo: Union[np.ndarray, torch.Tensor],
    dt: float = 0.002,
    max_samples: Optional[int] = None,
    pause_time: float = 0.5,
    save_path: Optional[str] = None,
    show_metrics: bool = True,
    interactive: bool = True,
):
    """
    Plota amostras sísmicas (real vs sintética) junto com wavelets estimadas.

    Parâmetros:
    -----------
    real : np.ndarray ou torch.Tensor
        Array de sinais reais com shape (n_amostras, n_amostras_sinal)
    syn : np.ndarray ou torch.Tensor
        Array de sinais sintéticos com shape (n_amostras, n_amostras_sinal)
        (pode ter tamanhos variáveis de convolução, será alinhado)
    solo : np.ndarray ou torch.Tensor
        Array de wavelets com shape (n_amostras, n_amostras_wavelet)
    dt : float
        Intervalo de amostragem em segundos (default: 0.002)
    max_samples : int, optional
        Número máximo de amostras a plotar. Se None, plota todas.
    save_path : str, optional
        Caminho para salvar as figuras. Se None, apenas exibe.
    show_metrics : bool
        Se True, exibe RMSE e correlação entre sinais reais e sintéticos
    """
    # Converter para numpy
    real = _to_numpy(real)
    syn = _to_numpy(syn)
    solo = _to_numpy(solo)

    # Garantir formato 2D
    real = _garantir_shape_2d(real, "real")
    syn = _garantir_shape_2d(syn, "syn")
    solo = _garantir_shape_2d(solo, "solo")

    # Alinhar tamanhos
    real, syn, solo = _alinhar_tamanhos_amostras(real, syn, solo)

    n_samples = real.shape[0]
    n_to_plot = min(n_samples, max_samples or n_samples)

    # Ativar modo interativo
    if interactive:
        plt.ion()

    # Criar figura uma única vez
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Inicializar linhas vazias
    (line_real,) = ax1.plot([], [], color="blue", lw=1.2, label="Real")
    (line_syn,) = ax1.plot([], [], color="red", lw=1.2, ls="--", label="Gerada")
    (line_erro,) = ax1.plot([], [], color="black", lw=0.8, alpha=0.4, label="Erro")
    (line_wavelet,) = ax2.plot([], [], color="blue", lw=1.5)
    vline_wavelet = ax2.axvline(0, color="k", linestyle=":", alpha=0.6)

    # Configurar eixos
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Amplitude (normalizado)")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True)
    ax2.set_title("Wavelet estimada")

    plt.tight_layout()

    # Loop através das amostras
    for i in range(n_to_plot):
        # Extrair amostra i
        real_i = real[i]
        syn_i = syn[i]
        solo_i = solo[i]

        # Garantir mesmo tamanho
        min_len = min(len(real_i), len(syn_i))
        real_i = real_i[:min_len]
        syn_i = syn_i[:min_len]

        # Vetores de tempo
        t_s = np.arange(len(real_i)) * dt
        t_w = (np.arange(len(solo_i)) - len(solo_i) // 2) * dt

        # Calcular erro
        erro = real_i - syn_i

        # --- ATUALIZAR SINAL ---
        line_real.set_data(t_s, real_i)
        line_syn.set_data(t_s, syn_i)
        line_erro.set_data(t_s, erro)

        # Ajustar limites do eixo
        ax1.set_xlim(t_s.min(), t_s.max())
        y_min = min(real_i.min(), syn_i.min(), erro.min())
        y_max = max(real_i.max(), syn_i.max(), erro.max())
        margin = (y_max - y_min) * 0.1
        ax1.set_ylim(y_min - margin, y_max + margin)

        # Título com métricas
        titulo = f"Sinal sísmico – Amostra {i}/{n_to_plot - 1}"
        if show_metrics:
            metricas = _calcular_metricas(real_i, syn_i)
            titulo += (
                f" (RMSE: {metricas['rmse']:.4f}, Corr: {metricas['correlacao']:.4f})"
            )
        ax1.set_title(titulo)

        # --- ATUALIZAR WAVELET ---
        w = solo_i / (np.max(np.abs(solo_i)) + 1e-8)
        line_wavelet.set_data(t_w, w)
        ax2.set_xlim(t_w.min(), t_w.max())

        # Salvar frame se necessário
        if save_path:
            fig.savefig(
                f"{save_path}_amostra_{i:03d}.png", dpi=100, bbox_inches="tight"
            )

        # Atualizar display
        if interactive:
            plt.pause(pause_time)
        else:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(pause_time)

    # Desativar modo interativo e manter figura aberta
    if interactive:
        plt.ioff()
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
    plt.plot(real, color='blue', alpha=0.7)
    plt.plot(syn, color='red', linestyle='--', alpha=0.7)
    plt.suptitle(f'Plot')
    plt.xlabel('Escala Unitária (índice dentro da amostra)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(['Real', 'Gerada'], loc='upper right')

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


def plot_axis(x, y):
    for i in range(x.shape[0]):
        plt.plot(x, y[i])
        plt.show()


def apply_ormsby_frequency_domain(spectrum, freq_axis, points=[5, 10, 60, 80]):
    """
    Aplica um filtro trapezoidal (Ormsby) diretamente em um espectro de amplitude.
    
    Parâmetros:
    - amplitude_spectrum: array 1D com as amplitudes do sinal.
    - freq_axis: array 1D com as frequências correspondentes a cada amplitude (eixo X).
    - points: lista [a, b, c, d] definindo os cantos do filtro em Hz.
    
    Retorno:
    - filtered_spectrum: O espectro de amplitude filtrado.
    - mask: O desenho do filtro (vetor de 0 a 1) para visualização.
    """
    a, b, c, d = points
    
    # Previne divisão por zero se o usuário colocar rampas verticais (b=a ou d=c)
    epsilon = 1e-10 
    
    # Criação da máscara (filtro) inicializada com zeros
    mask = np.zeros_like(spectrum)
        
    # 1. Rampa de Subida (a < f <= b)
    # Fórmula da reta: (f - a) / (b - a)
    idx_up = (freq_axis > a) & (freq_axis <= b)
    mask[..., idx_up] = (freq_axis[idx_up] - a) / (b - a + epsilon)
    
    # 2. Platô / Pass-band (b < f <= c)
    # Valor é 1.0 constante
    idx_pass = (freq_axis > b) & (freq_axis <= c)
    mask[..., idx_pass] = 1.0
    
    # 3. Rampa de Descida (c < f <= d)
    # Fórmula da reta descendo: 1 - (f - c) / (d - c)
    idx_down = (freq_axis > c) & (freq_axis <= d)
    mask[..., idx_down] = 1.0 - (freq_axis[idx_down] - c) / (d - c + epsilon)
        
    # Aplicação do filtro (Multiplicação ponto a ponto)
    filtered_spectrum = spectrum * mask
    
    return filtered_spectrum, mask