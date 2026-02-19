# 🍄 Super-FIA-Bros (NEAT Pipeline)

Questo branch (`NEAT_model`) contiene l'implementazione completa dell'agente basato sull'algoritmo genetico **NEAT (NeuroEvolution of Augmenting Topologies)** per il completamento del livello 1-1 di Super Mario Bros.

---

## 🚀 Guida alla Riproducibilità

L'intero processo di training, validazione e visualizzazione è stato progettato per essere eseguito su **Google Colab** per garantire la massima compatibilità e facilità d'uso.

### 1. 📂 Entry Point
Il file principale per riprodurre il progetto è il notebook:
👉 **[SuperMarioBros_Colab.ipynb](SuperMarioBros_Colab.ipynb)**

### 2. ▶️ Come Eseguire
1.  Apri il file `SuperMarioBros_Colab.ipynb` caricandolo su Google Colab o aprendolo localmente se hai un ambiente Jupyter con GPU.
2.  **Esegui tutte le celle in sequenza**. Il notebook si occuperà automaticamente di:
    *   Clonare questo repository.
    *   Installare le dipendenze corrette (incluso il downgrade di `numpy` necessario per `gym-super-mario-bros`).
    *   Scaricare i checkpoint pre-addestrati.

### 3. 🏋️‍♂️ Modalità di Esecuzione
All'interno del notebook (e tramite gli script nella cartella `src/`) sono disponibili tre modalità principali:

*   **Training da Zero:**
    Avvia una nuova evoluzione partendo dalla generazione 0.
    ```bash
    python src/main.py train --gen 100 --level 1-1
    ```

*   **Continuous Training (Consigliato):**
    Riprende l'addestramento dall'ultimo checkpoint salvato, mantenendo la "memoria" della specie.
    ```bash
    python src/cont_train.py
    ```

*   **Replay del Campione:**
    Visualizza e salva in video (`.mp4`) la performance del miglior genoma (il "vincitore").
    ```bash
    python src/replay_actions.py --level 1-1
    ```

---

## 🧠 Struttura del Progetto

Il codice sorgente è organizzato nella cartella `src/`:

| File | Descrizione |
| :--- | :--- |
| `src/config` | File di configurazione NEAT (iperparametri, mutazioni, popolazione). |
| `src/train.py` | Logica principale del training (valutazione genomi, parallelizzazione). |
| `src/cont_train.py` | Script per riprendere il training da checkpoint esistenti. |
| `src/replay_actions.py` | Genera video delle migliori run con overlay HUD. |
| `src/visualize.py` | Genera grafici dell’andamento del fitness e topologia della rete. |
| `src/main.py` | Wrapper principale per gestire i comandi da riga di comando. |

---

## 🌿 Struttura dei Branch

*   **`main` / `NEAT_model`**: Contiene l'implementazione NEAT stabile e completa (questo branch).

---

## 👥 Autori

*   [Luca Afeltra](https://github.com/luca-afe)
*   [Matteo De Stasio](https://github.com/Matteo-d-s)
*   [Marianna Diograzia](https://github.com/Erym35)
