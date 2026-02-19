# Super Mario Bros RL – PPO Agent

## 📋 Descrizione

Questo progetto implementa un agente di **Reinforcement Learning (RL)** capace di imparare a giocare a **Super Mario Bros (NES)** utilizzando l’algoritmo **PPO (Proximal Policy Optimization)**.

Il progetto è configurato per funzionare su **Windows 11**, superando le limitazioni di compatibilità delle librerie `nes-py` e `gym` tramite un ambiente **Python 3.11** specifico.


## ⚙️ Prerequisiti

Realizzare un agente in grado di **completare il livello 1-1 di Super Mario Bros**, confrontando due pipeline algoritmiche differenti e analizzandone:

### Visual Studio Build Tools

Necessari per compilare i componenti C++ dell’emulatore.
Durante l’installazione selezionare il carico di lavoro:

* **Sviluppo desktop con C++**

### Python 3.11.x

### 2️⃣ Attivazione

Attiva l'ambiente virtuale:

```powershell
.\mario_311\Scripts\activate
```

---

### 3️⃣ Installazione Dipendenze

L’ordine di installazione è **critico** per evitare conflitti su Windows. Esegui i comandi in sequenza:

#### A. Setup compilatori e compatibilità

```powershell
pip install setuptools==65.5.0 wheel<0.40.0
```

#### B. Emulatore e ambiente di gioco

```powershell
pip install nes-py
pip install gym_super_mario_bros==7.4.0
```

#### C. Librerie di Reinforcement Learning

```powershell
pip install gymnasium stable-baselines3[extra] shimmy
```

* distanza percorsa sull’asse orizzontale (x_pos)
* completamento del livello (bandiera finale)
* andamento dell’apprendimento (TensorBoard per PPO)
* evoluzione della fitness media (`avg_fitness.svg` per NEAT)
* costi computazionali e tempo di training

L’analisi completa è riportata nella documentazione.

---

## 📁 Struttura della repository (branch `main`)

```text
Super-FIA-Bros/
├── README.md
├── docs/
└── notebooks/
```

Le implementazioni specifiche dei modelli sono separate nei branch dedicati.

---

## 🧠 Struttura del Training

Una volta aperto Jupyter Lab, creare un nuovo notebook selezionando il kernel **Python (Mario 3.11)**.
Il flusso di lavoro è suddiviso in quattro celle logiche:

* **Import**: caricamento delle librerie (`gym`, `stable_baselines3`, `cv2`).
* **Preprocessing (Wrappers)**: conversione in scala di grigi (84×84).
* **Frame Stacking**: utilizzo di 4 frame consecutivi per percepire movimento e velocità.
* **Definizione Modello**: utilizzo di PPO (`CnnPolicy`) con iperparametri ottimizzati.

Setup cartelle per i log:

```python
tensorboard_log = "./logs/"
```

Training loop:

```python
model.learn(total_timesteps=1000000)
```

(con salvataggio periodico dei checkpoint)

---

## 📈 TensorBoard

Per visualizzare i grafici di apprendimento (aumento del Reward, diminuzione della Loss, ecc.) in tempo reale, mentre L'IA si allena:

1. Aprire un nuovo terminale PowerShell (lasciando quello del training in esecuzione);
2. Attivare l’ambiente virtuale;
3. Eseguire il comando puntando alla cartella dei log:

```powershell
tensorboard --logdir=./logs/
```

Aprire il browser all’indirizzo indicato (solitamente `http://localhost:6006`).

---

## 🛑 Risoluzione Problemi Comuni

**Errore: “Microsoft Visual C++ 14.0 is required”**
Verificare di aver installato i Build Tools (punto 1 dei Prerequisiti).

Per esplorare le implementazioni:

1. Passare al branch di interesse:

   * `PPO_model`
   * `NEAT_model`
2. Seguire le istruzioni presenti nei file e nei notebook del branch selezionato.

---

## ℹ️ Note Finali

* [Luca Afeltra](https://github.com/luca-afe)
* [Matteo De Stasio](https://github.com/Matteo-d-s)
* [Marianna Diograzia](https://github.com/Erym35)
