# Super Mario Bros RL – PPO Agent

## 📋 Descrizione

Questo progetto implementa un agente di **Reinforcement Learning (RL)** capace di imparare a giocare a **Super Mario Bros (NES)** utilizzando l’algoritmo **PPO (Proximal Policy Optimization)**.

Il progetto è configurato per funzionare su **Windows 11**, superando le limitazioni di compatibilità delle librerie `nes-py` e `gym` tramite un ambiente **Python 3.11** specifico.


## ⚙️ Prerequisiti

Prima di configurare l’ambiente virtuale, assicurati di avere installato:

### Visual Studio Build Tools

Necessari per compilare i componenti C++ dell’emulatore.
Durante l’installazione selezionare il carico di lavoro:

* **Sviluppo desktop con C++**

### Python 3.11.x

* Le versioni più recenti (3.12 / 3.13) **non sono compatibili** con `nes-py`
* Si consiglia l’installazione in un percorso semplice (es. `C:\Python311`)

### PowerShell Execution Policy

Abilitare l’esecuzione degli script aprendo PowerShell come amministratore ed eseguendo:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🛠️ Installazione e Setup

### 1️⃣ Creazione dell’Ambiente Virtuale

Poiché nel sistema potrebbero essere presenti più versioni di Python, viene forzato l’uso di Python 3.11. Sostituisci il percorso con quello della tua installazione.

```powershell
cd C:\Users\nome_utente\Desktop\marioia
"C:\Percorso\A\Python311\python.exe" -m venv mario_311
```

---

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

#### D. Jupyter Lab e Kernel

```powershell
pip install jupyterlab ipykernel
```

---

## 🚀 Avvio dell’Ambiente di Sviluppo

Per lavorare comodamente via browser con i Notebook interattivi, registra il Kernel (per renderlo visibile a Jupyter):

### Registrazione del Kernel

```powershell
python -m ipykernel install --user --name=mario_311 --display-name "Python (Mario 3.11)"
```

### Avvio Jupyter Lab

```powershell
python -m jupyterlab
```

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

**Errore import `gym_super_mario_bros`**
Assicurarsi di aver installato prima `nes-py` e poi `gym_super_mario_bros`.

**Jupyter non trova le librerie**
Controllare in alto a destra nel notebook che il kernel selezionato sia impostato su **Python (Mario 3.11)** e non quello globale **Python 3 (Global)**.

---

## ℹ️ Note Finali

Progetto configurato e sviluppato su **Windows 11**.
Ultimo aggiornamento: **Gennaio 2026**.

---
