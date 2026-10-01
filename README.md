# sense-lab

Entorno multidisciplinar en Python para:

1. **Procesamiento Avanzado de Señales** (Signal Processing)
2. **Monitorización de Salud Estructural** (Structure Health Monitoring, SHM)
3. **Predicción Cuantitativa de Mercados** (Quantitative Finance)

Este repositorio organiza **módulos** de descomposición modal (EMD, EEMDAN, ICEEMDAN) y herramientas auxiliares. Los métodos de filtrado de ruido están ahora en repositorios independientes.

---

## 📂 Estructura del Proyecto

```
sense-lab/
├─ README.md               ← Este fichero
├─ LICENSE                 ← Derechos reservados y cláusula de garantía
├─ requirements.txt        ← Dependencias globales
├─ signal processing/
│  ├─ decomposition/       ← Métodos de descomposición modal
│  │  ├─ emd/
│  │  │  └─ README.md      ← Documentación de EMD
│  │  ├─ eemdan/
│  │  │  └─ README.md      ← Documentación de EEMDAN
│  │  └─ iceemdan/
│  │     ├─ ICEEMDAN.py    ← Implementación CEEMDAN
│  │     ├─ requirements.txt
│  │     └─ README.md      ← Documentación de ICEEMDAN
└─ docs/                   ← (Opcional) Documentación adicional
```

---

## 📦 Instalación

1. Clona el repositorio:

   ```bash
   git clone git@github.com:KimiK1983/sense-lab.git
   cd sense-lab
   ```

2. Crea un entorno virtual y actívalo:

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\\Scripts\\activate     # Windows PowerShell
   ```

3. Instala dependencias globales:

   ```bash
   pip install -r requirements.txt
   ```

4. Para submódulos específicos, instala sus requerimientos:

   ```bash
   cd signal\ processing\decomposition\iceemdan
   pip install -r requirements.txt
   ```

---

## 🚀 Uso Rápido

### Descomposición Modal (EMD / EEMDAN / ICEEMDAN)

```python
from signal_processing.decomposition.emd import EMD
from signal_processing.decomposition.eemdan import EEMDAN
from signal_processing.decomposition.iceemdan import CEEMDAN

# Ejemplo sintético
time, noisy_signal, clean_components = ...  # Genera señal compleja

# 1. EMD básico
emd = EMD(); imfs = emd.emd(noisy_signal)

# 2. EEMDAN
#eemdan = EEMDAN(trials=100, epsilon=0.2)
imfs_eemdan = EEMDAN(trials=100, epsilon=0.2)(noisy_signal)

# 3. ICEEMDAN (paralelo)
ice = CEEMDAN(trials=150, epsilon=0.25, parallel=True)
components = ice(noisy_signal)
```

### Filtrado de Ruido (Denoising)

- [BayesShrink denoising](https://github.com/KimiK1983/bayesshrink-denoising)
- [NeighShrink-SURE denoising](https://github.com/KimiK1983/neighshrink-sure-denoising)

---

## 🛠️ Requisitos

> **requirements.txt (global)**

```
numpy>=1.20.0
matplotlib
pandas  # si usas análisis tabular
EMD-signal>=0.2.4  # PyEMD oficial
tqdm>=4.60.0
```

> **iceemdan/requirements.txt**  (submódulo ICEEMDAN)

```
EMD-signal>=0.2.4
numpy>=1.20.0
tqdm>=4.60.0
```

---

## 📖 Licencias y Terceros

- **Código Propietario**: Todos los derechos reservados © 2025 Javier F. Santamaria.\
  Véase `LICENSE` en la raíz para la cláusula de garantía y responsabilidad.

- **Terceros**:

  - **EMD-signal (PyEMD)** – Apache License 2.0. Licencia completa en `licenses/APACHE-2.0.txt`.\
    Fuente: [https://github.com/laszukdawid/PyEMD](https://github.com/laszukdawid/PyEMD)
  - **tqdm**, **numpy**, **matplotlib** – Licencias MIT/BSD.

---

## 🤝 Contribuciones

1. Haz un *fork* del repositorio.
2. Crea una rama de características: `git checkout -b feature/nueva-funcion`.
3. Haz *commit* de tus cambios: `git commit -m "Añade nueva función"`.
4. Envía tu rama a tu fork: `git push origin feature/nueva-funcion`.
5. Abre un *Pull Request* describiendo tus cambios.

Por favor, mantén el estilo de código y añade tests cuando sea posible.

---

## 📬 Contacto

Javier F. Santamaria\
✉️ [javier.santamaria\_mex@gmail.com](mailto\:javier.santamaria_mex@gmail.com)

