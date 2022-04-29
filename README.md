# replication-topic-modelling-in-embedding-space
**Installieren aller benötigten Packages**

`pip install -r requirements.txt`

**Reposititory**
1. Implementierung von Vorverarbeitungen und Modellen im Ordner [src](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/tree/main/src)
2. Ausgaben jedes Schritts von den durchgeführten Experimenten in der Datei [notebook_replication.ipynb](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_replication.ipynb) 

**Ziel**
1. Replizieren des Artikels: [Topic Modelling in embedding space](https://arxiv.org/abs/1907.04907) von Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei. Originales [Code](https://github.com/adjidieng/ETM) von Autoren. 
2. Neues Experiment: Kombination von pre-fitted BERT-Wordembedding mit ETM

**Datensatz**
1. [20NewsGroups](), [New York Times]()
2. Vorverarbeitungsschritten: 
3. Traindatensatz, Testdatensatz (Testdatensatz-h1, Testdatensatz-h2) und Validationssatz von 100 Dokumenten

**ETM-Modell**
1. Das Vocabular besteht aus den einzigartigen Wörtern aus dem Traindatensatz
2. Eingabedaten: BOW-Repräsentation für jedes Dokumentes des Datensatzes (doc={(word-id, word-frequency)})

**Experimenten**

- [ ] Vergleich Top-5-Wörter von den top 7 meisten genutzen Topics (Datensatz: 1.8M Documents von NYT, Corpus V=212237, K=300) zwischen LDA und ETM
- [ ] Vergleich zwischen LDA und ETM auf dem Datensatz: 20NewsGroups (Maß: Topic Quality = Topic Coherence * Topic Diversity, Predictive Performance)
- [ ] Ergebnisse von ETM auf NYT-Stopwords und NYT-ohne-Stopwords
- [ ] Vergleich der Embedding zwischen CBOW und BERT-Wortembedding
- [ ] Vergleich zwischen prefitted-CBOW/SKIPGramm-ETM und prefitted-BERT-ETM

**Dokumentation für Teamarbeit**
- Einpaar Befehlen für die Nutzung von Google Colab für Teamarbeit:
1. !git config --global user.email ""
2. !git config --global user.name ""
3. !git clone https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space.git
4. cd /content/replication-topic-modelling-in-embedding-space
5. !git add /content/replication-topic-modelling-in-embedding-space
6. !git commit -m "test git push from google colab"
7. !git remote rm origin
8. !git remote add origin https://<username>:<token>@github.com/<username>/replication-topic-modelling-in-embedding-space.git
9. !git push --set-upstream origin main
10. !git pull
