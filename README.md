# Super-FIA-Bros

## 📋 Descrizione

**Super-FIA-Bros** è un progetto sviluppato per il corso di **Fondamenti di Intelligenza Artificiale** che analizza e confronta due approcci di IA applicati a un ambiente dinamico e sequenziale: **Super Mario Bros** (livello 1-1), emulato tramite il framework `gym-super-mario-bros`.

Il progetto studia il comportamento di un **agente intelligente** che, a partire da input visivo e con un insieme di azioni discrete, deve avanzare nel livello fino al suo completamento.

---

## 🎯 Obiettivo

L’obiettivo del progetto è la realizzazione di un agente in grado di **completare il livello 1-1 di Super Mario Bros**, confrontando due pipeline algoritmiche differenti e analizzandone:

* prestazioni
* stabilità dell’apprendimento
* costi computazionali
* trade-off tra approcci

---

## 🧠 Pipeline implementate

### Pipeline 1 – Deep Reinforcement Learning (PPO)

* Algoritmo: Proximal Policy Optimization (PPO)
* Policy: CNN (CnnPolicy)
* Input: frame di gioco preprocessati (grayscale, resize, frame stacking)
* Reward basata su esplorazione, progresso nel livello e penalità

I dettagli tecnici e le istruzioni di esecuzione sono disponibili nella cartella `notebooks/IA_1` (branch `ia_1`).

---

### Pipeline 2 – Neuroevoluzione (NEAT)

* Algoritmo: NEAT (NeuroEvolution of Augmenting Topologies)
* Evoluzione di pesi e topologia della rete neurale
* Fitness basata sulla distanza percorsa (x_pos)
* Speciazione, elitismo e stagnazione configurati esplicitamente

I dettagli tecnici e le istruzioni di esecuzione sono disponibili nella cartella `notebooks/IA_2` (branch `ia_2`).

---

## ℹ️ Nota sugli algoritmi

Gli algoritmi adottati (PPO e NEAT) non sono trattati direttamente nel programma del corso, ma rappresentano un’estensione coerente dei concetti fondamentali affrontati, quali agente in ambiente sequenziale, funzione di prestazione (reward/fitness), esplorazione e valutazione sperimentale con analisi dei trade-off.

---

## 📊 Valutazione e Trade-off

Le pipeline vengono confrontate utilizzando metriche comuni, tra cui:

* distanza percorsa sull’asse orizzontale (x_pos)
* completamento del livello (bandiera finale)
* andamento dell’apprendimento (TensorBoard per PPO, `avg_fitness.svg` per NEAT)
* costi computazionali e tempo di training

I risultati e i grafici finali sono riportati nella documentazione.

---

## 📁 Struttura della repository

```text
Super-FIA-Bros/
├── README.md
├── docs/
├── rl_model/
├── neat_model/
└── notebooks/
    ├── IA_1/
    └── IA_2/
```

---

## 📄 Documentazione

La documentazione completa del progetto (definizione del problema, specifica PEAS, descrizione delle pipeline, preprocessing, valutazione e conclusioni) è disponibile nella cartella:

```
docs/
```

---

## ▶️ Riproducibilità

Per riprodurre gli esperimenti:

1. selezionare la pipeline di interesse (`IA_1` o `IA_2`)
2. seguire le istruzioni riportate nei notebook e nei file di supporto presenti nelle rispettive cartelle

---

## 👥 Autori

* [Luca Afeltra](https://github.com/luca-afe)
* [Matteo De Stasio](https://github.com/Matteo-d-s)
* [Marianna Diograzia](https://github.com/Erym35)

---
