#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
archivo: neighshrink_sure.py
descripción: Implementación de referencia de un algoritmo de denoising de señales
             basado en Wavelets, combinando las técnicas NeighShrink, SURE
             (Stein's Unbiased Risk Estimate) y Cycle-Spinning.
             El código está optimizado para alto rendimiento con Numba.

autor: Javier F. Santamaria <javier.santamaria_mex@gmail.com>
fecha: 2025-06-13
licencia: © 2025 Javier F. Santamaria. Todos los derechos reservados.
           Véase el fichero LICENSE en la raíz del proyecto.
"""

import time
import pywt
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List

# --- 1. CONFIGURACIÓN DE OPTIMIZACIÓN CON NUMBA ---
# El rendimiento de este algoritmo depende críticamente de la compilación
# Just-In-Time (JIT) proporcionada por la biblioteca Numba.

try:
    from numba import njit, prange
except ImportError:
    # Proporciona un fallback funcional si Numba no está instalado, permitiendo
    # que el código se ejecute (lentamente) para fines de depuración.
    print("*" * 80)
    print("ADVERTENCIA: La biblioteca 'Numba' no está instalada.")
    print("El código se ejecutará en modo de puro Python, lo cual será extremadamente lento.")
    print("Para un rendimiento óptimo, instale Numba con: pip install numba")
    print("*" * 80)
    def njit(parallel: bool = False):
        def decorator(func):
            return func
        return decorator
    prange = range

# --- 2. FUNCIONES NÚCLEO DEL ALGORITMO DE DENOISING ---

# Las siguientes funciones constituyen el corazón del método de denoising.
# Están diseñadas para ser modulares, eficientes y teóricamente rigurosas.

@njit(parallel=True, cache=True)
def _calculate_S_periodic(sub_coef_sq: np.ndarray, Rsim: int) -> np.ndarray:
    """
    Calcula la energía del vecindario local (S_k^2) para cada coeficiente.

    Esta es una operación de suma móvil sobre los coeficientes al cuadrado.
    La implementación utiliza padding periódico explícito para mantener la
    consistencia teórica con la Transformada Wavelet Discreta (DWT) en modo
    'periodization' y la técnica de Cycle-Spinning.

    Args:
        sub_coef_sq (np.ndarray): Array 1D de los coeficientes de detalle al cuadrado.
        Rsim (int): Radio del vecindario (la ventana total es 2*Rsim + 1).

    Returns:
        np.ndarray: Un array de la misma longitud que la entrada, donde cada
                    elemento S[k] contiene la suma de la energía en el
                    vecindario del k-ésimo coeficiente.
    """
    Ns = sub_coef_sq.shape[0]
    S = np.zeros_like(sub_coef_sq)
    # Este bucle está paralelizado por Numba para una ejecución eficiente en CPUs multinúcleo.
    for k in prange(Ns):
        s_val = 0.0
        for i in range(-Rsim, Rsim + 1):
            # El operador módulo (%) implementa el padding periódico (wrap-around).
            s_val += sub_coef_sq[(k + i) % Ns]
        S[k] = s_val
    return S

@njit(cache=True)
def _find_best_threshold_for_rsim(sub_coef: np.ndarray, Rsim: int) -> Tuple[float, float]:
    """
    Encuentra el umbral óptimo (λ) para un radio de vecindario (Rsim) dado,
    minimizando el Riesgo No Sesgado de Stein (SURE).

    SURE proporciona una estimación insesgada del error cuadrático medio (MSE),
    lo que permite seleccionar un umbral adaptativo a los datos sin necesidad
    de conocer la señal limpia original.

    Args:
        sub_coef (np.ndarray): Coeficientes de detalle normalizados por la
                               desviación estándar del ruido.
        Rsim (int): Radio del vecindario.

    Returns:
        Tuple[float, float]: Una tupla conteniendo (riesgo_mínimo, umbral_óptimo).
    """
    Ns = sub_coef.shape[0]
    sub_coef_sq = sub_coef**2

    S = _calculate_S_periodic(sub_coef_sq, Rsim)
    # Estabilidad numérica para evitar la división por cero.
    S[S == 0] = 1e-9

    # Rango de umbrales candidatos a evaluar.
    Thres = np.arange(Rsim + 1.0, (Rsim + 1.0) * 3.01, 0.1)
    risks = np.zeros(Thres.shape[0])

    # Bucle secuencial sobre los umbrales.
    for i, th in enumerate(Thres):
        th2 = th**2
        
        is_hard = S <= th2
        is_soft = ~is_hard

        # Inicialización de términos de riesgo.
        term1, term2, term3 = 0.0, 0.0, 0.0
        
        # Cálculo vectorizado de los términos de la fórmula de riesgo SURE.
        # Se añaden guardas para manejar robustamente el caso de arrays vacíos.
        if np.any(is_soft):
            S_soft, d2_soft = S[is_soft], sub_coef_sq[is_soft]
            S2_soft = S_soft**2
            term1 = np.sum(1.0 / S_soft - 2.0 * d2_soft / S2_soft)
            term2 = np.sum(d2_soft / S2_soft)

        if np.any(is_hard):
            d2_hard = sub_coef_sq[is_hard]
            term3 = np.sum(d2_hard) - 2.0 * d2_hard.shape[0]
        
        risks[i] = Ns - 2.0 * th2 * term1 + th2 * th2 * term2 + term3

    ibest = np.argmin(risks)
    return risks[ibest], Thres[ibest]

def _subband_thresholding(sub_coef: np.ndarray, attenuation_factor: float = 1.0) -> np.ndarray:
    """
    Aplica el umbralado NeighShrink a una sub-banda de coeficientes.

    Esta función orquesta la búsqueda del par (Rsim, umbral) óptimo y luego
    aplica la regla de encogimiento (shrinkage) a los coeficientes.

    Args:
        sub_coef (np.ndarray): Coeficientes de detalle a filtrar.
        attenuation_factor (float): Factor heurístico (<= 1.0) para relajar
                                    el umbral, útil en niveles de descomposición
                                    más gruesos.

    Returns:
        np.ndarray: Los coeficientes de la sub-banda filtrados.
    """
    Rsim_set = [1, 2, 3]
    best_risk, best_threshold, Rsim_opt = float('inf'), 0.0, 1

    # 1. Búsqueda del par (Rsim, λ) que minimiza globalmente el riesgo SURE.
    for Rsim in Rsim_set:
        risk, thres = _find_best_threshold_for_rsim(sub_coef, Rsim)
        if risk < best_risk:
            best_risk, best_threshold, Rsim_opt = risk, thres, Rsim

    final_threshold = best_threshold * attenuation_factor
    
    # 2. Aplicación de la regla de encogimiento de NeighShrink.
    sub_coef_sq = sub_coef**2
    S = _calculate_S_periodic(sub_coef_sq, Rsim_opt)
    S[S == 0] = 1e-9

    # Factor de encogimiento: β_k = max(0, 1 - λ²/S_k²)
    factor = 1.0 - (final_threshold**2) / S
    factor = np.maximum(factor, 0.0)
    
    return sub_coef * factor

def _estimate_noise_sigma(detail_coeffs: np.ndarray) -> float:
    """
    Estima la desviación estándar del ruido (σ) de la señal.

    Utiliza el estimador robusto de Mediana de la Desviación Absoluta (MAD)
    sobre los coeficientes de detalle del nivel más fino de la DWT (cD1),
    donde se asume que la energía de la señal es mínima.

    Args:
        detail_coeffs (np.ndarray): Coeficientes de la sub-banda cD1.

    Returns:
        float: La desviación estándar estimada del ruido.
    """
    if len(detail_coeffs) == 0:
        return 0.0
    # La constante 0.6745 asegura la consistencia asintótica del estimador
    # para ruido Gaussiano.
    return np.median(np.abs(detail_coeffs)) / 0.6745

def _level_dependent_denoising_base(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Función base de denoising para una única señal (sin Cycle-Spinning).

    Aplica la DWT, estima el ruido, y realiza el umbralado NeighShrink-SURE
    en cada nivel de detalle de forma adaptativa.
    """
    wavelet = pywt.Wavelet(params['wavelet_name'])
    level = params['level']
    mode = 'periodization'
    
    coeffs = pywt.wavedec(data, wavelet, level=level, mode=mode)
    noise_sigma = _estimate_noise_sigma(coeffs[-1])
    
    if noise_sigma == 0: return data

    reconstructed_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        current_level_detail = level - i + 1
        
        # Heurística de atenuación: relaja el umbral en los niveles más gruesos,
        # donde la señal es más prominente, logrando un mejor equilibrio entre
        # la eliminación de ruido y la preservación de detalles.
        att_factor = 1.0
        if current_level_detail >= params['attenuation_start_level']:
            att_factor = params['attenuation_factor']
            
        norm_coeffs = coeffs[i] / noise_sigma
        thresholded_norm = _subband_thresholding(norm_coeffs, attenuation_factor=att_factor)
        reconstructed_coeffs.append(thresholded_norm * noise_sigma)
        
    return pywt.waverec(reconstructed_coeffs, wavelet, mode=mode)[:len(data)]

# --- 3. ORQUESTADOR PRINCIPAL DEL DENOISING ---

def neighshrink_sure_denoise(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Aplica el algoritmo de denoising completo, incluyendo la técnica de
    Cycle-Spinning para lograr invarianza a la traslación.

    Cycle-Spinning promedia los resultados de aplicar el denoising a múltiples
    versiones desplazadas cíclicamente de la señal, mitigando eficazmente los
    artefactos que la DWT puede introducir.

    Args:
        data (np.ndarray): La señal 1D de entrada a filtrar.
        params (Dict[str, Any]): Diccionario de parámetros de configuración, que debe
                                 incluir 'wavelet_name', 'level', 'shifts', etc.

    Returns:
        np.ndarray: La señal filtrada.
    """
    n = len(data)
    wavelet = pywt.Wavelet(params['wavelet_name'])
    
    # Lógica adaptativa para el número de shifts en señales largas.
    # El coste del Cycle-Spinning es lineal con el número de shifts. Para
    # señales muy largas, un número alto de shifts puede ser prohibitivo.
    # El mínimo teórico para asegurar la invarianza es la longitud del filtro.
    default_shifts = 32
    if n > 8192: # Umbral empírico para considerar una señal como "larga"
        shifts = min(default_shifts, wavelet.dec_len)
        print(f"INFO: Señal larga detectada (N={n}). Reduciendo shifts a {shifts} por eficiencia.")
    else:
        shifts = params.get('shifts', default_shifts)
    
    if shifts == 0:
        return _level_dependent_denoising_base(data, params)
        
    denoised_sum = np.zeros(n, dtype=np.float64)
    print(f"INFO: Iniciando Denoising con NeighShrink-SURE (shifts={shifts})...")
    
    start_time = time.perf_counter()
    for i in range(shifts):
        # Desplazar -> Filtrar -> Deshacer desplazamiento
        shifted_data = np.roll(data, i)
        denoised_shifted = _level_dependent_denoising_base(shifted_data, params)
        denoised_sum += np.roll(denoised_shifted, -i)
    end_time = time.perf_counter()
    
    print(f"INFO: Denoising completado en {end_time - start_time:.2f} segundos.")
    return denoised_sum / shifts

# --- 4. EJECUCIÓN Y DEMOSTRACIÓN ---

def _create_benchmark_signal(name: str, n_points: int) -> np.ndarray:
    """Carga una de las señales de benchmark de PyWavelets."""
    try:
        return pywt.data.demo_signal(name, n=n_points)
    except Exception as e:
        print(f"ERROR: No se pudo cargar la señal de benchmark '{name}'. Error: {e}")
        return None

def run_benchmark_experiment(signal_name: str, noise_level: float, n_points: int = 1024):
    """
    Ejecuta un experimento de denoising en una señal sintética con referencia (ground truth)
    y calcula métricas como el SNR.
    """
    print("\n" + "#" * 70)
    print(f"#      BENCHMARK CON LA SEÑAL: '{signal_name.upper()}'      #")
    print("#" * 70)
    
    signal_clean = _create_benchmark_signal(signal_name, n_points)
    if signal_clean is None: return

    N = len(signal_clean)
    t = np.linspace(0, 1, N, endpoint=False)
    
    np.random.seed(42)
    noise_sigma_val = noise_level * np.std(signal_clean)
    noise = np.random.normal(0, noise_sigma_val, N)
    signal_noisy = signal_clean + noise

    denoising_params = {
        'wavelet_name': 'sym8', 'level': 5, 'shifts': 32,
        'attenuation_start_level': 2, 'attenuation_factor': 0.9
    }
    signal_denoised = neighshrink_sure_denoise(signal_noisy, params=denoising_params)

    # Cálculo de métricas con referencia
    snr_noisy = 10 * np.log10(np.sum(signal_clean**2) / np.sum((signal_clean - signal_noisy)**2))
    snr_denoised = 10 * np.log10(np.sum(signal_clean**2) / np.sum((signal_clean - signal_denoised)**2))

    # Visualización
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Benchmark de Denoising en '{signal_name}'", fontsize=16)
    axes[0].plot(t, signal_clean, 'k-', label='Original Limpia')
    axes[1].plot(t, signal_noisy, 'r-', alpha=0.7, label=f'Ruidosa (SNR: {snr_noisy:.2f} dB)')
    axes[2].plot(t, signal_denoised, 'g-', label=f'Filtrada (SNR: {snr_denoised:.2f} dB)')
    
    for ax in axes:
        ax.legend(loc='upper right'); ax.grid(True, linestyle=':'); ax.set_ylabel('Amplitud')
    axes[2].set_xlabel('Tiempo Normalizado')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
    
    print(f"\n--- Resultados del Benchmark '{signal_name.upper()}' ---")
    print(f"Mejora total en SNR: {snr_denoised - snr_noisy:+.2f} dB")


def _precompile_numba_functions():
    """
    Fuerza la compilación JIT de las funciones Numba antes del primer uso real
    para evitar el retardo inicial (calentamiento).
    """
    print("INFO: Pre-compilando funciones JIT (calentamiento)...")
    start_warmup = time.perf_counter()
    dummy_array = np.random.randn(100).astype(np.float64)
    _find_best_threshold_for_rsim(dummy_array, Rsim=1)
    end_warmup = time.perf_counter()
    print(f"INFO: Calentamiento completado en {end_warmup:.2f} segundos.")


if __name__ == '__main__':
    
    # 1. Calentar las funciones JIT para un rendimiento consistente.
    _precompile_numba_functions()
    
    # 2. Ejecutar benchmarks en señales sintéticas estándar.
    run_benchmark_experiment('doppler', noise_level=0.4, n_points=1024)
    run_benchmark_experiment('blocks', noise_level=0.1, n_points=1024)
    run_benchmark_experiment('bumps', noise_level=0.2, n_points=1024)
    run_benchmark_experiment('heavisine', noise_level=0.1, n_points=1024)
    
    # Aquí es donde integrarías la carga y el procesamiento de tus datos reales,
    # similar al script 'compare_denoising_methods.py'.
    # Ejemplo:
    #
    # from your_data_loader import load_my_real_signal
    # my_signal, fs = load_my_real_signal()
    # params = {'wavelet_name': 'sym8', 'level': 7, ...}
    # denoised_my_signal = neighshrink_sure_denoise(my_signal, params)
    # plt.plot(denoised_my_signal)
    # plt.show()