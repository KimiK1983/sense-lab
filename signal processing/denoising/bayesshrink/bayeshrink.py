#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
archivo: bayes_shrink_denoising.py
descripción: Implementación de denoising con DWT y BayesShrink.
autor: Javier F. Santamaria <javier.santamaria_mex@gmail.com>
fecha: 2025-06-12
licencia: © 2025 Javier F. Santamaria. Todos los derechos reservados.
           Véase el fichero LICENSE en la raíz del proyecto.
"""

import pywt
import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FUNCIONES AUXILIARES Y DE ESTIMACIÓN
# ==============================================================================

def _estimate_noise_sigma(detail_coeffs):
    """
    Estima la desviación estándar del ruido (sigma) a partir de los coeficientes
    de detalle de la sub-banda de mayor frecuencia (cD1).

    Esta función utiliza el robusto estimador de la Mediana de la Desviación
    Absoluta (MAD).

    Parámetros:
    -----------
    detail_coeffs : array-like
        Array de coeficientes de detalle (normalmente, la primera sub-banda cD1).

    Retorna:
    --------
    float
        La desviación estándar estimada del ruido (sigma).
    """
    # La constante 0.6745 es el factor de consistencia para una distribución normal.
    # Es el cuantil 0.75 de la distribución normal estándar.
    # sigma = median(|cD1|) / 0.6745
    if len(detail_coeffs) == 0:
        return 0.0
    
    # np.median es más eficiente y robusto que el cálculo manual
    median_abs_coeffs = np.median(np.abs(detail_coeffs))
    
    return median_abs_coeffs / 0.6745

def _bayes_shrink_threshold(detail_coeffs, noise_sigma):
    """
    Calcula el umbral óptimo para una sub-banda de detalle usando el método
    BayesShrink.

    Parámetros:
    -----------
    detail_coeffs : array-like
        Array de coeficientes de detalle de la sub-banda actual.
    noise_sigma : float
        La desviación estándar del ruido (previamente estimada).

    Retorna:
    --------
    float
        El valor del umbral calculado para la sub-banda.
    """
    # Convertir a array de numpy para operaciones vectorizadas eficientes
    detail_coeffs = np.asarray(detail_coeffs)
    
    # 1. Calcular la varianza de la sub-banda de detalle (sigma_y^2)
    #    varianza = E[X^2] - (E[X])^2.
    #    Se asume que la media de los coeficientes de detalle es cero.
    variance_total = np.mean(detail_coeffs**2)
    
    # 2. Estimar la varianza de la señal limpia (sigma_x^2)
    #    sigma_x^2 = max(0, sigma_y^2 - sigma_n^2)
    noise_variance = noise_sigma**2
    signal_variance = max(0, variance_total - noise_variance)

    # 3. Calcular el umbral de BayesShrink
    #    Si la varianza de la señal es cero, la sub-banda es puro ruido.
    #    El umbral debe ser infinito para anular todos los coeficientes.
    #    Se usa el máximo valor absoluto como umbral práctico en este caso.
    if signal_variance == 0:
        return np.max(np.abs(detail_coeffs)) if detail_coeffs.size > 0 else 0
    
    #    T_B = sigma_n^2 / sigma_x
    threshold = noise_variance / math.sqrt(signal_variance)
    
    return threshold

# ==============================================================================
# 2. FUNCIÓN PRINCIPAL DE DENOISING
# ==============================================================================

def bayes_shrink_denoising(data, wavelet_name='sym8', level=None, mode='soft'):
    """
    Realiza el filtrado de ruido (denoising) de una señal 1D utilizando la
    transformada wavelet discreta y el método de umbralización BayesShrink.

    El proceso sigue los siguientes pasos analíticos:
    1.  Descomposición Wavelet: La señal se descompone en varios niveles para
        separar los coeficientes de aproximación (bajas frecuencias) de los
        coeficientes de detalle (altas frecuencias).
    2.  Estimación de Varianza del Ruido: Se calcula una única estimación
        robusta de la varianza del ruido a partir de la sub-banda de detalle
        de más alta frecuencia (cD1), donde se asume que la señal tiene poca
        energía.
    3.  Umbralización Adaptativa por Sub-banda: Para cada sub-banda de detalle,
        se calcula un umbral óptimo usando BayesShrink. Este umbral depende
        tanto de la varianza del ruido global como de la varianza de la señal
        local en esa sub-banda.
    4.  Aplicación del Umbral: El umbral calculado se aplica a los coeficientes
        de detalle utilizando una función de umbral (típicamente 'soft').
    5.  Reconstrucción Wavelet: La señal filtrada se reconstruye a partir de los
        coeficientes de aproximación originales y los coeficientes de detalle
        ya umbralizados.

    Parámetros:
    -----------
    data : array-like
        La señal 1D de entrada que se desea filtrar.
    wavelet_name : str, opcional
        El nombre de la wavelet a utilizar (p. ej., 'db4', 'sym8').
        'sym8' es una buena elección por su simetría y soporte.
        Default: 'sym8'.
    level : int, opcional
        El nivel de descomposición wavelet. Si es None, se calcula
        automáticamente el máximo nivel posible. Default: None.
    mode : {'soft', 'hard'}, opcional
        El tipo de umbralización a aplicar. 'soft' (soft-thresholding)
        generalmente produce resultados visualmente más agradables al no
        crear discontinuidades. Default: 'soft'.

    Retorna:
    --------
    numpy.ndarray
        La señal filtrada.
    """
    # --- Verificación de Entradas ---
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("La entrada 'data' debe ser un array o lista.")
    
    # --- CORRECCIÓN APLICADA AQUÍ ---
    if wavelet_name not in pywt.wavelist(kind='all'):
        raise ValueError(f"La wavelet '{wavelet_name}' no existe en PyWavelets. Wavelets disponibles: {pywt.wavelist(kind='all')}")
        
    data = np.asarray(data)
    wavelet = pywt.Wavelet(wavelet_name)

    # --- 1. Descomposición Wavelet ---
    # Si no se especifica el nivel, calcular el máximo nivel posible
    if level is None:
        level = pywt.dwt_max_level(data_len=len(data), filter_len=wavelet.dec_len)
        print(f"Nivel de descomposición no especificado. Usando nivel máximo: {level}")

    # Realizar la descomposición wavelet discreta (DWT)
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # coeffs es una lista: [cAn, cDn, cDn-1, ..., cD1]
    # cA: coeficientes de aproximación, cD: coeficientes de detalle
    
    # --- 2. Estimación de Varianza del Ruido ---
    # Se utiliza la sub-banda de detalle de mayor frecuencia (cD1), que es coeffs[-1]
    noise_sigma = _estimate_noise_sigma(coeffs[-1])
    if noise_sigma == 0:
        print("Advertencia: La desviación estándar del ruido estimada es cero. La señal puede no tener ruido o ser demasiado corta.")
        # Si no hay ruido, no es necesario hacer nada, devolvemos la señal original.
        return data

    # --- 3. y 4. Umbralización Adaptativa por Sub-banda ---
    # Iterar sobre las sub-bandas de detalle para aplicar el umbral.
    # No se umbraliza el coeficiente de aproximación (coeffs[0]).
    reconstructed_coeffs = [coeffs[0]] # Mantener cA sin cambios
    for i in range(1, len(coeffs)):
        detail_coeffs = coeffs[i]
        # Calcular el umbral específico para esta sub-banda
        threshold = _bayes_shrink_threshold(detail_coeffs, noise_sigma)
        
        # Aplicar el umbral a los coeficientes de detalle
        thresholded_detail_coeffs = pywt.threshold(detail_coeffs, threshold, mode=mode)
        reconstructed_coeffs.append(thresholded_detail_coeffs)

    # --- 5. Reconstrucción Wavelet ---
    # Reconstruir la señal a partir de los nuevos coeficientes.
    denoised_data = pywt.waverec(reconstructed_coeffs, wavelet)
    
    # Asegurar que la señal de salida tenga la misma longitud que la de entrada
    return denoised_data[:len(data)]

# ==============================================================================
# 3. EJEMPLO DE USO Y VISUALIZACIÓN
# ==============================================================================

if __name__ == '__main__':
    # --- a. Generación de una señal de prueba ---
    # Crear una señal compuesta por bloques y saltos
    try:
        # La función demo_signal puede estar en diferentes ubicaciones según la versión
        signal_clean = pywt.data.demo_signal('blocks', n=1024)
    except AttributeError:
        # Fallback para versiones más antiguas de PyWavelets
        from pywt.data import demo_signal
        signal_clean = demo_signal('blocks', n=1024)
        
    N = len(signal_clean)
    t = np.linspace(0, 1, N, endpoint=False)
    
    # Añadir ruido gaussiano
    noise_std_dev = 0.25
    noise = np.random.normal(0, noise_std_dev, N)
    signal_noisy = signal_clean + noise

    # --- b. Aplicación del filtro BayesShrink ---
    # Llamar a la función principal de denoising
    signal_denoised = bayes_shrink_denoising(
        data=signal_noisy,
        wavelet_name='sym8',
        level=5,
        mode='soft'
    )

    # --- c. Cálculo de métricas de rendimiento ---
    # Error Cuadrático Medio (MSE) y Relación Señal-Ruido (SNR)
    mse_noisy = np.mean((signal_clean - signal_noisy)**2)
    mse_denoised = np.mean((signal_clean - signal_denoised)**2)
    
    snr_noisy = 10 * np.log10(np.sum(signal_clean**2) / np.sum((signal_clean - signal_noisy)**2))
    snr_denoised = 10 * np.log10(np.sum(signal_clean**2) / np.sum((signal_clean - signal_denoised)**2))

    # --- d. Visualización de resultados ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Denoising con Wavelet y BayesShrink', fontsize=16)

    axes[0].plot(t, signal_clean, color='darkblue')
    axes[0].set_title('Señal Original Limpia')
    axes[0].set_ylabel('Amplitud')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(t, signal_noisy, color='red', alpha=0.8)
    axes[1].set_title(f'Señal Ruidosa (MSE: {mse_noisy:.4f}, SNR: {snr_noisy:.2f} dB)')
    axes[1].set_ylabel('Amplitud')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    axes[2].plot(t, signal_denoised, color='green')
    axes[2].set_title(f'Señal Filtrada (MSE: {mse_denoised:.4f}, SNR: {snr_denoised:.2f} dB)')
    axes[2].set_xlabel('Tiempo')
    axes[2].set_ylabel('Amplitud')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n--- Resultados del Denoising ---")
    print(f"Desviación estándar del ruido añadida: {noise_std_dev}")
    print(f"MSE de la señal ruidosa: {mse_noisy:.4f}")
    print(f"MSE de la señal filtrada: {mse_denoised:.4f}")
    print(f"SNR de la señal ruidosa: {snr_noisy:.2f} dB")
    print(f"SNR de la señal filtrada: {snr_denoised:.2f} dB")
    print(f"Mejora en SNR: {snr_denoised - snr_noisy:.2f} dB")