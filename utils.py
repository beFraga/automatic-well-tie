import matplotlib.pyplot as plt
import numpy as np
import torch


def normalization(x):
    return x / (torch.max(x, dim=-1, keepdim=True).values + 1e-8)


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
        ax1.plot(real[i], color='blue', alpha=0.7)
        ax1.plot(syn[i], color='red', linestyle='--', alpha=0.7)
        ax1.set_title(f'Plot')
        ax1.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend(['Real', 'Gerada'], loc='upper right')


        ax2.plot(solo[i])
        ax2.set_title(f'')
        ax2.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        plt.tight_layout()

        plt.show()


def adjust_data_length(data, target_length=300, device='cpu'):
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
    ax.plot(x, color='blue', alpha=0.7)
    ax.set_title(f'Plot')
    ax.set_xlabel('Escala Unitária (índice dentro da amostra)')
    ax.set_ylabel('Amplitude')
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
        ax1.set_title(f'Plot {i}')
        ax1.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)


        ax2.plot(b[i])
        ax2.set_title(f'')
        ax2.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax2.set_ylabel('Amplitude')
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
        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 6))

        ax1.plot(a[i])
        ax1.set_title(f'Plot {i}')
        ax1.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        ax2.plot(b[i])
        ax2.set_title(f'')
        ax2.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        ax3.plot(c[i])
        ax3.set_title(f'')
        ax3.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True)

        ax4.plot(d[i])
        ax4.set_title(f'')
        ax4.set_xlabel('Escala Unitária (índice dentro da amostra)')
        ax4.set_ylabel('Amplitude')
        ax4.grid(True)

        plt.tight_layout()

        plt.show()


def plot_axis(x, y):
    if len(y.shape) > 1:
        for i in range(y.shape[0]):
            plt.plot(x, y[i])
            plt.show()
    else:
        plt.plot(x, y)
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
    spectrum = spectrum.detach().numpy()
    
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
    
    return torch.tensor(filtered_spectrum, dtype=torch.float32), mask


def r_coefficient(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(
        np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2)
    )

    return numerator / denominator