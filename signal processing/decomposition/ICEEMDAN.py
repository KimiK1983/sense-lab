#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Basado en PyEMD (Apache 2.0) – https://github.com/laszukdawid/PyEMD
"""
archivo: iceemdan.py
descripción: Extensión Improved CEEMDAN de PyEMD (ICEEMDAN) basada en Colominas et al. (2014).
autor: Javier F. Santamaria <javier.santamaria_mex@gmail.com>
fecha: 2025-06-16
licencia: © 2025 Javier F. Santamaria. Todos los derechos reservados.
           Véase el fichero LICENSE en la raíz del proyecto.
"""

import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
from tqdm import tqdm

# Define the multiprocessing wrapper at the module level for pickling
def _emd_decomposer_mp_friendly(args_tuple: Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]) -> np.ndarray:
    """
    MP-friendly EMD decomposition function.
    Instantiates EMD within the worker process.
    """
    signal_to_decompose, time_vector, emd_instance_params = args_tuple
    # Late import to avoid issues if EMD itself has complex module-level setup
    # Ensure PyEMD.EMD is accessible in the environment where workers run.
    from PyEMD import EMD

    # Create a new EMD instance in each worker process
    emd_worker_instance = EMD(**emd_instance_params)
    return emd_worker_instance.emd(signal_to_decompose, time_vector)


class CEEMDAN:
    """
    **"Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise"**
    (Colominas, Schlotthauer, Torres, 2014)

    This implementation is based on the original PyEMD CEEMDAN, modified
    to incorporate improvements proposed by:
    M.A. Colominas, G. Schlotthauer, M.E. Torres, "Improved complete
    ensemble EMD: A suitable tool for biomedical signal processing",
    In Biomed. Sig. Proc. and Control, V. 14, 2014, pp. 19--29.

    The key improvements from the paper are:
    1. Defining IMFs as the difference between the current residue and an
       average of local means, aiming to reduce noise in IMFs.
    2. Using the k-th IMF of white noise (E_k(w)) to help extract the
       k-th signal IMF, aiming to reduce spurious modes.

    Parameters
    ----------
    trials : int (default: 100)
        Number of trials (ensemble size) for noise addition.
    epsilon : float (default: 0.05)
        The noise amplitude coefficient (ε₀ in Colominas et al., 2014).
        A common value, also suggested for original CEEMDAN, is around 0.2.
    ext_EMD : EMD instance or None (default: None)
        An external EMD object can be passed. If None, a default EMD instance is created.
        If using `parallel=True` with `ext_EMD`, it's recommended to also pass `emd_params`
        to ensure worker processes correctly re-instantiate EMD.
    parallel : bool (default: False)
        Whether to use multiprocessing for trial computations.
    processes : int or None (optional)
        Number of processes to use if `parallel` is True. Defaults to `cpu_count()`.
    noise_kind : str (default: "normal")
        Type of base white noise $w^{(i)}$ to generate. Allowed: "normal" (Gaussian)
        or "uniform". The theory in Colominas et al. (2014) is based on Gaussian noise.
        Using "uniform" noise is considered experimental in this context.
    range_thr : float or None (default: None)
        Optional stopping criterion for CEEMDAN: if the range (max - min) of the
        normalized residue falls below this threshold, decomposition stops.
        To replicate Colominas et al. (2014) strictly, set to `None`.
    total_power_thr : float or None (default: None)
        Optional stopping criterion for CEEMDAN: if the total power (sum of absolute values)
        of the normalized residue falls below this threshold, decomposition stops.
        To replicate Colominas et al. (2014) strictly, set to `None`.
    residue_std_threshold : float (default: 1e-7)
        Optional stopping criterion for CEEMDAN: if the standard deviation of the
        normalized residue falls below this threshold, decomposition stops. This helps
        prevent attempts to decompose near-flat residues.
    seed : int or None (default: None)
        Seed for the random number generator for reproducible results.
    max_imf_iterations : int (default: 100)
        Hard limit on the number of IMFs to extract by CEEMDAN. This acts as a safeguard.
        The paper's algorithm does not specify such a limit for the number of IMFs.
    dtype : numpy.dtype (default: np.float64)
        Data type to be used for internal computations. Using `np.float32`
        can reduce memory footprint for very long signals, at the cost of precision.
    emd_params : dict or None (default: None)
        Parameters to pass to the `EMD.__init__` method. This is used when `ext_EMD`
        is `None` (to create the internal EMD instance) and critically when `parallel=True`
        (for re-instantiating EMD in worker processes).
        Example: `emd_params={'MAX_ITERATION': 1000, 'spline_kind': 'akima'}`.

    Notes on Memory Usage
    ---------------------
    This implementation pre-decomposes all noise realizations (`trials` times) and stores
    all their IMFs (`E_k(w^(i))`). For very long signals or a high number of `trials`,
    this can lead to significant memory consumption during the
    `_pre_decompose_noise_for_ceemdan` phase and while these IMFs are held.
    Memory for these noise IMFs is released at the end of the `ceemdan` method call.
    """

    logger = logging.getLogger(__name__)
    noise_kinds_all = ["normal", "uniform"]

    def __init__(self, trials: int = 100, epsilon: float = 0.05, ext_EMD=None, parallel: bool = False,
                 emd_params: Optional[Dict[str, Any]] = None, dtype: Any = np.float64, **kwargs):

        self.trials = trials
        self.epsilon = epsilon
        self.range_thr = kwargs.get("range_thr", None)
        self.total_power_thr = kwargs.get("total_power_thr", None)
        self.residue_std_threshold = float(kwargs.get("residue_std_threshold", 1e-7))
        self.dtype = dtype

        self.random = np.random.RandomState(seed=kwargs.get("seed"))
        self.noise_kind = kwargs.get("noise_kind", "normal")
        if self.noise_kind not in self.noise_kinds_all:
            raise ValueError(f"Unsupported noise kind: {self.noise_kind}. Allowed: {self.noise_kinds_all}")
        if self.noise_kind == "uniform":
            self.logger.warning(
                "Using 'uniform' noise. The theory in Colominas et al. (2014) is "
                "based on Gaussian ('normal') noise. Uniform noise behavior is experimental."
            )

        self.max_imf_iterations = int(kwargs.get("max_imf_iterations", 100))
        self.parallel = parallel

        # Procesamiento del parámetro 'processes'
        user_processes_input = kwargs.get("processes")
        if user_processes_input is None or \
           (isinstance(user_processes_input, int) and user_processes_input <= 0):
            # Si es None, o es un entero <= 0, usar cpu_count()
            # Pool(processes=0) o Pool(processes=-1) pueden dar error o comportamientos no deseados.
            self.processes = cpu_count()
            if isinstance(user_processes_input, int) and user_processes_input <= 0 and user_processes_input is not None:
                self.logger.info(
                    f"Value '{user_processes_input}' for 'processes' is invalid or implies default. Using cpu_count() = {self.processes}."
                )
        elif not isinstance(user_processes_input, int):
             self.logger.warning(
                f"Invalid type '{type(user_processes_input)}' for 'processes'. Expected int or None. Using cpu_count() = {cpu_count()}."
            )
             self.processes = cpu_count()
        else: # Es un entero positivo
            self.processes = user_processes_input

        # Advertencia si se especifican procesos pero no se usa el modo paralelo
        if user_processes_input is not None and not self.parallel:
            # Solo advertir si el usuario realmente intentó especificar un número
            self.logger.warning(
                f"Passed value for 'processes' ({user_processes_input}) has no effect when 'parallel' is False."
            )

        self.all_noise_IMFs_for_CEEMDAN: List[np.ndarray] = []
        self.avg_std_E1_w = 1.0 # Default, será calculado

        # Parámetros para la instancia EMD interna y para los workers en modo paralelo
        self.emd_params_internal = emd_params.copy() if emd_params is not None else {}
        self.emd_params_internal.setdefault('DTYPE', self.dtype) # Asegurar que DTYPE se pasa a EMD

        if ext_EMD is None:
            from PyEMD import EMD # Asegurar que PyEMD.EMD está disponible
            self.EMD = EMD(**self.emd_params_internal)
        else:
            self.EMD = ext_EMD
            # Si se usa ext_EMD y modo paralelo, es bueno tener emd_params para los workers
            # La instancia self.EMD proporcionada se usará en el proceso principal si no es paralelo
            if self.parallel and not emd_params: # Si no se proveyeron emd_params explícitamente
                 self.logger.warning(
                     "Parallel mode with 'ext_EMD': consider providing 'emd_params' "
                     "for robust EMD re-instantiation in worker processes. "
                     "Attempting to infer parameters from the provided ext_EMD instance."
                 )
                 try: # pragma: no cover
                     # Intenta obtener parámetros públicos y simples de la instancia EMD
                     inferred_params = {
                         k: v for k, v in self.EMD.__dict__.items()
                         if isinstance(v, (int, float, str, bool, tuple)) and not k.startswith('_')
                     }
                     # Actualiza emd_params_internal con los inferidos, pero no sobrescribas los ya existentes
                     # (aunque en este caso emd_params era None, así que no hay conflicto)
                     for key, val in inferred_params.items():
                         self.emd_params_internal.setdefault(key, val)
                     self.emd_params_internal.setdefault('DTYPE', self.dtype) # Re-asegurar DTYPE
                 except AttributeError: # pragma: no cover
                     self.logger.error(
                        "Could not infer parameters from ext_EMD for parallel workers. "
                        "Default EMD parameters (plus DTYPE specified for CEEMDAN) will be used in workers."
                     )
        
        self.C_IMF: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __call__(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress: bool = False) -> np.ndarray:
        """
        Performs the Improved CEEMDAN decomposition.

        Parameters
        ----------
        S : np.ndarray
            Input signal (1D).
        T : Optional[np.ndarray] (default: None)
            Time vector for the signal. Not directly used in core CEEMDAN logic but passed to EMD.
        max_imf : int (default: -1)
            Maximum number of IMFs to extract. If -1, uses criteria from paper and `max_imf_iterations`.
        progress : bool (default: False)
            Whether to display a progress bar for trial loops.
        
        Returns
        -------
        np.ndarray
            Array containing all extracted IMFs and the final residue as the last row.
        """
        return self.ceemdan(S, T=T, max_imf=max_imf, progress=progress)

    def __getstate__(self) -> Dict: # pragma: no cover
        self_dict = self.__dict__.copy()
        if "pool" in self_dict: del self_dict["pool"]
        # For general pickling of the CEEMDAN instance, we might not want to pickle
        # self.EMD if it's complex or contains C extensions.
        # However, _emd_decomposer_mp_friendly handles EMD creation in workers.
        # If self.EMD is an instance of PyEMD.EMD, it should be picklable.
        return self_dict

    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Generates white noise with a specified standard deviation (`scale`).
        """
        if self.noise_kind == "normal":
            noise = self.random.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind == "uniform":
            # For U(-A, A), variance is A^2/3. Std dev is A/sqrt(3).
            # If desired std dev is `scale`, then A/sqrt(3) = `scale` => A = scale * sqrt(3).
            limit = scale * np.sqrt(3)
            noise = self.random.uniform(low=-limit, high=limit, size=size)
        else: # Should not be reached due to check in __init__
            raise ValueError(f"Internal error: Unsupported noise kind {self.noise_kind}") # pragma: no cover
        return noise.astype(self.dtype)

    def noise_seed(self, seed: int) -> None: # pragma: no cover
        """Set seed for the internal random number generator."""
        self.random.seed(seed)

    def _get_emd_local_mean(self, S_plus_noise: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
        """Helper to get M(S_plus_noise) = S_plus_noise - E_1(S_plus_noise) using EMD."""
        imfs = self.EMD.emd(S_plus_noise.astype(self.dtype), T, max_imf=1)
        # EMD.emd returns [imf1, residue_after_imf1] or [signal] if no imf1.
        # The local mean is the residue after sifting out the first IMF.
        return imfs[-1].astype(self.dtype)

    def _pre_decompose_noise_for_ceemdan(self, S_shape: tuple, T: Optional[np.ndarray]):
        """
        Generates 'trials' realizations of unit variance white noise w^(i)
        and decomposes each into E_k(w^(i)). Stores them.
        Also calculates the average standard deviation of the first noise IMF, <std(E₁(w))>.
        """
        self.logger.info("Generating and decomposing all noise realizations for CEEMDAN (this may take time)...")
        
        # Generar todas las realizaciones de ruido base w^(i) (varianza unitaria, dtype configurado)
        all_w_realizations = [self.generate_noise(scale=1.0, size=S_shape) for _ in range(self.trials)]

        # Determinar si la barra de progreso tqdm debe estar deshabilitada
        tqdm_disabled = not (self.logger.isEnabledFor(logging.INFO) or self.logger.isEnabledFor(logging.DEBUG))
        desc = "Decomposing base noise w^(i)"
        
        if self.parallel:
            # Preparar argumentos para el wrapper _emd_decomposer_mp_friendly
            # self.emd_params_internal ya contiene DTYPE y otros parámetros EMD
            args_for_pool = [(noise, T, self.emd_params_internal) for noise in all_w_realizations]
            
            try:
                # Usar un context manager para Pool asegura que se cierre correctamente
                with Pool(processes=self.processes) as pool:
                    self.all_noise_IMFs_for_CEEMDAN = list(tqdm(pool.imap(_emd_decomposer_mp_friendly, args_for_pool),
                                                          total=self.trials, desc=desc, disable=tqdm_disabled))
            except Exception as e: # Capturar excepciones durante el procesamiento en paralelo
                self.logger.error(f"Error during parallel EMD of noise realizations: {e}")
                self.logger.error(
                    "This might be due to issues with pickling, EMD parameters for workers, "
                    "or an error within the EMD process in a worker."
                )
                self.logger.info("Consider running in non-parallel mode (parallel=False) for more detailed error messages or to isolate the issue.")
                # Re-lanzar la excepción para que el usuario sepa que algo falló críticamente
                # O, alternativamente, se podría intentar un fallback a modo no paralelo aquí,
                # pero eso podría ocultar problemas subyacentes.
                raise e # Propagar la excepción
        else:
            # Modo no paralelo: descomponer secuencialmente
            # La instancia self.EMD ya está configurada con los parámetros correctos (incluyendo DTYPE)
            self.all_noise_IMFs_for_CEEMDAN = [self.EMD.emd(noise, T) for noise in tqdm(all_w_realizations, desc=desc, disable=tqdm_disabled)]
        
        # Asegurar que los IMFs de ruido resultantes tengan el dtype configurado
        # Esto es importante porque EMD podría devolver un dtype diferente si no se configuró DTYPE internamente.
        self.all_noise_IMFs_for_CEEMDAN = [imfs.astype(self.dtype) for imfs in self.all_noise_IMFs_for_CEEMDAN]
        
        # Calcular la desviación estándar promedio del primer IMF de ruido, <std(E₁(w))>
        std_E1_w_sum = 0.0
        valid_E1_count = 0
        for noise_imfs_trial in self.all_noise_IMFs_for_CEEMDAN:
            if len(noise_imfs_trial) > 0 and len(noise_imfs_trial[0]) > 0: # Verificar si E₁(w) existe y no está vacío
                std_E1_w_sum += np.std(noise_imfs_trial[0])
                valid_E1_count +=1
        
        if valid_E1_count == 0: # pragma: no cover
             self.logger.warning(
                 "No first IMFs (E1(w)) found for any noise realization. "
                 "Using default avg_std_E1_w = 1.0. Beta_0 calculation may be affected."
             )
             self.avg_std_E1_w = 1.0
        else:
            self.avg_std_E1_w = std_E1_w_sum / valid_E1_count
        
        if self.avg_std_E1_w == 0: # pragma: no cover
            # Esto es muy improbable si valid_E1_count > 0 y los E1(w) no son planos
            self.logger.warning(
                "Calculated average std of E1(w) is zero. Using 1.0 to avoid division by zero in beta_0. "
                "This might indicate an issue with noise generation or EMD of noise."
            )
            self.avg_std_E1_w = 1.0

    def ceemdan(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1, progress: bool = False) -> np.ndarray:
        S_orig_dtype = S.dtype
        S = S.astype(self.dtype).ravel()

        S_std_orig = np.std(S)
        if S_std_orig == 0:
            self.logger.warning("Signal standard deviation is zero. Returning signal as the only component (residue).")
            self.C_IMF = np.empty((0, S.size), dtype=self.dtype)
            self.residue = S.copy()
            return S.reshape(1, -1).astype(S_orig_dtype)

        S_scaled = S / S_std_orig # Normalize signal: std(S_scaled) = 1

        self._pre_decompose_noise_for_ceemdan(S_scaled.shape, T)

        extracted_cimfs_list: List[np.ndarray] = []
        # current_residue represents r_k from the paper (or x_norm for the first step)
        # At start of loop for d_k, current_residue = r_{k-1} (or x_norm if k=1)
        current_residue = S_scaled.copy() 

        # Stage for d_1 (first cIMF)
        # β₀ = ε₀ * std(S_scaled) / <std(E₁(w))>. Since std(S_scaled)=1, β₀ = ε₀ / <std(E₁(w))>.
        beta_0 = self.epsilon / self.avg_std_E1_w
        
        self.logger.debug(f"Calculating 1st cIMF (d_1) with beta_0 = {beta_0:.4f}")
        sum_local_means_stage0 = np.zeros_like(S_scaled, dtype=self.dtype)
        
        pbar_trials_stage0 = tqdm(range(self.trials), desc="cIMF 1 (d_1)", disable=not progress)
        num_valid_trials_stage0 = 0
        for trial_idx in pbar_trials_stage0:
            if len(self.all_noise_IMFs_for_CEEMDAN[trial_idx]) > 0: # E_1(w^(i)) exists
                E1_w = self.all_noise_IMFs_for_CEEMDAN[trial_idx][0]
                signal_plus_noise = current_residue + beta_0 * E1_w # x_norm + β₀*E₁(w^(i))
                sum_local_means_stage0 += self._get_emd_local_mean(signal_plus_noise, T)
                num_valid_trials_stage0 +=1

        if num_valid_trials_stage0 == 0: # pragma: no cover
            self.logger.error("No valid trials for the first cIMF. Cannot proceed. Returning original signal.")
            # Ensure attributes are set before returning, even if just to empty/original state
            self.C_IMF = np.empty((0, S.size), dtype=self.dtype)
            self.residue = S.copy() # S is already self.dtype here
            return S.reshape(1, -1).astype(S_orig_dtype)
            
        r1_avg = sum_local_means_stage0 / num_valid_trials_stage0 # This is r₁ from paper
        cIMF1 = current_residue - r1_avg                        # This is d₁ = x_norm - r₁
        
        extracted_cimfs_list.append(cIMF1)
        current_residue = r1_avg # current_residue is now r_1, for calculating d_2

        # Stages for d_k, k >= 2
        # Determine the limit for the number of IMFs to extract in the loop
        effective_max_imf = self.max_imf_iterations # Start with internal hard limit
        if max_imf > 0: # User specified a total number of IMFs for CEEMDAN
            effective_max_imf = max_imf

        # Loop to extract d_2, d_3, ..., up to d_{effective_max_imf}
        # imf_idx_1based is the 1-based index of the IMF being extracted (e.g., 2 for d_2)
        for imf_idx_1based in range(2, effective_max_imf + 1):
            self.logger.debug(f"Attempting to extract cIMF {imf_idx_1based} (d_{imf_idx_1based})")
            
            # Check end condition based on current_residue (which is r_{imf_idx_1based-1} from paper)
            # num_extracted_cimfs is len(extracted_cimfs_list), which is imf_idx_1based-1
            if self.end_condition(current_residue, max_imf, imf_idx_1based - 1):
                self.logger.debug(f"End condition met before extracting cIMF {imf_idx_1based}.")
                break

            std_current_residue = np.std(current_residue)
            if std_current_residue < self.residue_std_threshold:
                self.logger.debug(
                    f"Residue std ({std_current_residue:.2e}) is below threshold "
                    f"({self.residue_std_threshold}). Stopping."
                )
                break
            
            # For extracting d_k (here k = imf_idx_1based):
            # Residue is r_{k-1} (current_residue)
            # Noise is E_k(w) (from self.all_noise_IMFs_for_CEEMDAN at index k-1)
            # Beta is β_{k-1} = ε₀ * std(r_{k-1})
            noise_imf_list_idx = imf_idx_1based - 1 # 0-indexed for list access to E_k(w)
            beta_val = self.epsilon * std_current_residue # This is β_{k-1}
            
            self.logger.debug(
                f"Calculating cIMF {imf_idx_1based} (d_{imf_idx_1based}) "
                f"with beta_{imf_idx_1based-1} = {beta_val:.4f} "
                f"using E_{noise_imf_list_idx+1}(w)" # Log E_k with 1-based index
            )
            sum_local_means_stage_k = np.zeros_like(S_scaled, dtype=self.dtype)
            
            pbar_trials_stage_k = tqdm(range(self.trials), desc=f"cIMF {imf_idx_1based} (d_{imf_idx_1based})", disable=not progress)
            num_valid_trials_stage_k = 0
            for trial_idx in pbar_trials_stage_k:
                if len(self.all_noise_IMFs_for_CEEMDAN[trial_idx]) > noise_imf_list_idx:
                    Ek_w = self.all_noise_IMFs_for_CEEMDAN[trial_idx][noise_imf_list_idx] # E_k(w^(i))
                    signal_plus_noise = current_residue + beta_val * Ek_w # r_{k-1} + β_{k-1}*E_k(w^(i))
                    sum_local_means_stage_k += self._get_emd_local_mean(signal_plus_noise, T)
                    num_valid_trials_stage_k += 1
                # else: This noise realization didn't have enough IMFs. Handled by num_valid_trials_stage_k.

            if num_valid_trials_stage_k == 0: # pragma: no cover
                self.logger.warning(
                    f"No valid trials for cIMF {imf_idx_1based} (d_{imf_idx_1based}) "
                    f"due to insufficient noise IMFs E_{noise_imf_list_idx+1}(w). Stopping extraction."
                )
                break
            
            r_k_avg = sum_local_means_stage_k / num_valid_trials_stage_k # This is r_k from paper
            cIMF_k = current_residue - r_k_avg                         # This is d_k = r_{k-1} - r_k
            
            extracted_cimfs_list.append(cIMF_k)
            current_residue = r_k_avg # Update current_residue to r_k for the next iteration (to calculate d_{k+1})

        # After the loop, current_residue is the final residue r_K
        extracted_cimfs_list.append(current_residue)
        
        # Scale all components back to original signal's scale
        all_components = np.array(extracted_cimfs_list, dtype=self.dtype) * S_std_orig
        
        # Free memory used by pre-decomposed noise IMFs
        del self.all_noise_IMFs_for_CEEMDAN[:]
        self.all_noise_IMFs_for_CEEMDAN = [] # Ensure it's an empty list
        
        # Store results in instance attributes
        self.C_IMF = all_components[:-1]
        self.residue = all_components[-1]

        self.logger.info(f"CEEMDAN decomposition finished. Found {self.C_IMF.shape[0]} IMFs and 1 residue.")
        return all_components.astype(S_orig_dtype) # Return with original dtype

    def end_condition(self, current_residue_r_k: np.ndarray, 
                      max_imf_user_request: int, num_extracted_cimfs: int) -> bool:
        """
        Tests for end condition of CEEMDAN.
        Called with current_residue_r_k (which is r_j if we are about to extract d_{j+1})
        and num_extracted_cimfs (which is j).
        """
        # Check if user-requested max_imf has been reached
        if 0 < max_imf_user_request <= num_extracted_cimfs:
            self.logger.debug(f"Stopping: User-requested max_imf ({max_imf_user_request}) reached.")
            return True

        # Check if residue can be decomposed further by EMD
        # (i.e., it's not an IMF itself and has enough extrema)
        imfs_of_residue = self.EMD.emd(current_residue_r_k.astype(self.dtype), None, max_imf=2)
        if len(imfs_of_residue) <= 1:
            self.logger.debug("Stopping: Residue cannot be decomposed further by EMD (is IMF or <3 extrema).")
            return True
        
        # Optional thresholds (applied to normalized residue)
        if self.range_thr is not None and \
           (np.max(current_residue_r_k) - np.min(current_residue_r_k)) < self.range_thr:
            self.logger.debug(f"Stopping: Residue range below threshold ({self.range_thr}).")
            return True

        if self.total_power_thr is not None and \
           np.sum(np.abs(current_residue_r_k)) < self.total_power_thr:
            self.logger.debug(f"Stopping: Residue power below threshold ({self.total_power_thr}).")
            return True
            
        return False

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]: # pragma: no cover
        """
        Provides access to separated imfs and residue from recently analysed signal.
        """
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No IMFs found. Please, run CEEMDAN method first.")
        return self.C_IMF, self.residue

    # The emd method wrapper might be useful if users want to access the configured EMD instance directly
    # but it's not strictly necessary for CEEMDAN's operation.
    # def emd(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1) -> np.ndarray:
    #     """Provides access to the internal EMD instance's emd method."""
    #     return self.EMD.emd(S, T, max_imf=max_imf)