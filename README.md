
Super Mario Bros RL - PPO Agent
Questo progetto implementa un agente di Reinforcement Learning (RL) capace di imparare a giocare a Super Mario Bros (NES) utilizzando l'algoritmo PPO (Proximal Policy Optimization).
Il progetto è configurato per funzionare su Windows 11, superando le limitazioni di compatibilità delle librerie nes-py e gym tramite un ambiente Python 3.11 specifico.

Prerequisiti
Prima di configurare l'ambiente virtuale, assicurati di avere i seguenti componenti installati nel sistema:

1 Visual Studio Build Tools
  Necessario per compilare i componenti C++ dell'emulatore.
  Durante l'installazione, selezionare il carico di lavoro: "Sviluppo desktop con C++".
  Python 3.11.x
    Le versioni più recenti (es. 3.12/3.13) non sono compatibili con nes-py.
    Si consiglia di installarlo in un percorso semplice (es. C:\Python311)
    PowerShell Execution Policy
    Abilitare l'esecuzione degli script digitando in PowerShell (Amministratore):
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Installazione e Setup
Segui questi passaggi per ricreare l'ambiente di sviluppo isolato.

1. Creazione dell'Ambiente Virtuale
Poiché nel sistema potrebbero esserci altre versioni di Python, forziamo l'uso di Python 3.11.
Sostituisci il percorso con quello della tua installazione.
  cd C:\Users\nome_utente\Desktop\marioia
  & "C:\Percorso\A\Python311\python.exe" -m venv mario_311

2. Attivazione
Attiva l'ambiente virtuale:
  .\mario_311\Scripts\activate

3. Installazione Dipendenze
L'ordine di installazione è critico per evitare conflitti su Windows. Esegui i comandi in sequenza:
  A. Setup dei compilatori e compatibilità:
    pip install setuptools==65.5.0 wheel<0.40.0

  B. Emulatore e Ambiente di Gioco:
    pip install nes-py
    pip install gym_super_mario_bros==7.4.0

  C. Librerie di Reinforcement Learning (Stable Baselines 3):
    pip install gymnasium stable-baselines3[extra] shimmy

  D. Jupyter Lab e Kernel:
    pip install jupyterlab ipykernel

Avvio dell'Ambiente di Sviluppo
  Per lavorare comodamente via browser con i Notebook interattivi:
  Registra il Kernel (per renderlo visibile a Jupyter):
  python -m ipykernel install --user --name=mario_311 --display-name "Python (Mario 3.11)"

Avvia Jupyter Lab:
  python -m jupyterlab

Struttura del Training
  Una volta aperto Jupyter Lab, crea un nuovo notebook selezionando il kernel Python (Mario   3.11). Il flusso di lavoro consigliato è diviso in 4 celle logiche:

  Import: Caricamento delle librerie (gym, stable_baselines3, cv2).

  Preprocessing (Wrappers): Conversione in scala di grigi (84x84 pixel).

  Frame Stacking: L'IA riceve 4 frame consecutivi per percepire movimento e velocità.
  
  Definizione Modello: Utilizzo dell'algoritmo PPO (CnnPolicy).
  
  Iperparametri ottimizzati per stabilità.

  Setup cartelle per i log: tensorboard_log='./logs/'.

Training Loop: 
  Comando model.learn(total_timesteps=1000000).
  Salvataggio periodico dei checkpoint.

Monitoraggio Training (TensorBoard)

  Per visualizzare i grafici dell'apprendimento (aumento del Reward, diminuzione della Loss,  ecc.)   in tempo reale mentre l'IA si allena:
  Apri un nuovo terminale PowerShell (lascia quello del training in esecuzione).
  Attiva l'ambiente virtuale (come al punto 2 dell'installazione).
  Esegui il comando puntando alla cartella dei log:

tensorboard --logdir=./logs/
  Apri il browser all'indirizzo che apparirà (solitamente http://localhost:6006).

Risoluzione Problemi Comuni
  Errore "Microsoft Visual C++ 14.0 is required":
  Verifica di aver installato i Build Tools (punto 1 dei Prerequisiti).
  Errore nell'import di gym_super_mario_bros:
  Assicurati di aver installato prima nes-py e poi gym_super_mario_bros.
  Jupyter non trova le librerie:
  Controlla in alto a destra nel notebook che il kernel sia impostato su "Python (Mario 3.11)" e   non su "Python 3 (Global)".

Progetto configurato e sviluppato su Windows 11.
Ultimo aggiornamento: Gennaio 2026
