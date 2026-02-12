# Super-FIA-Bros

## 📋 Descrizione

**Super-FIA-Bros** è un progetto sviluppato per il corso di **Fondamenti di Intelligenza Artificiale** che analizza e confronta due approcci di IA applicati a un ambiente dinamico e sequenziale: **Super Mario Bros** (livello 1-1), emulato tramite il framework `gym-super-mario-bros`.

Il progetto studia il comportamento di un **agente intelligente** che, a partire da input visivo e con un insieme di azioni discrete, deve avanzare nel livello fino al suo completamento.

---

## 🎯 Obiettivo

Realizzare un agente in grado di **completare il livello 1-1 di Super Mario Bros**, confrontando due pipeline algoritmiche differenti e analizzandone:

* prestazioni
* stabilità dell’apprendimento
* costi computazionali
* trade-off tra approcci

---

## 🧠 Pipeline implementate

### 🔵 PPO – Deep Reinforcement Learning

* Algoritmo: Proximal Policy Optimization (PPO)
* Policy: CNN (CnnPolicy)
* Input: frame preprocessati (grayscale, resize, frame stacking)
* Reward progettata per incentivare avanzamento e completamento del livello

Il codice e i risultati relativi a questa pipeline sono disponibili nel branch:

```
PPO_model
```

---

### 🟣 NEAT – Neuroevoluzione

* Algoritmo: NEAT (NeuroEvolution of Augmenting Topologies)
* Evoluzione di pesi e topologia della rete neurale
* Fitness basata sulla distanza percorsa (x_pos)
* Speciazione, elitismo e stagnazione configurati esplicitamente

Questa pipeline è stata integrata e valutata sul livello 1-1.

Il codice e i risultati relativi a questa pipeline sono disponibili nel branch:

```
NEAT_model
```

---

## 📊 Valutazione e Trade-off

Le pipeline vengono confrontate utilizzando metriche comuni, tra cui:

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

## 📄 Documentazione

La documentazione completa del progetto (definizione del problema, specifica PEAS, descrizione delle pipeline, preprocessing, valutazione e conclusioni) è disponibile nella cartella:

```
docs/
```

---

## ▶️ Riproducibilità

Per esplorare le implementazioni:

1. Passare al branch di interesse:

   * `PPO_model`
   * `NEAT_model`
2. Seguire le istruzioni presenti nei file e nei notebook del branch selezionato.

---

## 👥 Autori

* [Luca Afeltra](https://github.com/luca-afe)
* [Matteo De Stasio](https://github.com/Matteo-d-s)
* [Marianna Diograzia](https://github.com/Erym35)
