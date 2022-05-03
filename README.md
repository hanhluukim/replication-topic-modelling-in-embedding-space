# replication-topic-modelling-in-embedding-space

**Ziel des Projekts**
1. Replizieren des Artikels: [Topic Modelling in embedding space](https://arxiv.org/abs/1907.04907) von Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei. Das originales [Code](https://github.com/adjidieng/ETM) von Autoren. 
2. Neues Experiment: Kombination von pre-fitted BERT-Embedding mit ETM

**Gebrauchte Pakette**

`pip install -r requirements.txt`

**Struktur des Repos**
1. Implementierung befindet sich im Ordner [src](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/tree/main/src)
2. Jeder Schritt von den betrachteten Experimenten kann in der Datei [notebook_replication.ipynb](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_replication.ipynb) kontrolliert werden. Man kann dieses Notebook selbst auf dem Link Colab öffnen und durchführen. 


**Genutzte Datensätze**
1. [20NewsGroups](), [New York Times]()
2. Vorverarbeitungsschritten: 
3. Traindatensatz, Testdatensatz (Testdatensatz-h1, Testdatensatz-h2) und Validationssatz von 100 Dokumenten

**Angewandte Word-Embedding Methoden**
- [ ] CBOW und Skipgram mittels [Gensim](https://radimrehurek.com/gensim/)
- [ ] BERT-Wordembedding mittels [Transformer-Huggingface](https://huggingface.co/docs/transformers/installation)

**Architektur des ETM-Modells**
1. Das Vocabular besteht aus den einzigartigen Wörtern aus dem Traindatensatz
2. Eingabedaten: BOW-Repräsentation für jedes Dokumentes des Datensatzes (doc={(word-id, word-frequency)})

**Durchgeführte Experimenten**

- [ ] Vergleich Top-5-Wörter von den top 7 meisten genutzen Topics (Datensatz: 1.8M Documents von NYT, Corpus V=212237, K=300) zwischen LDA und ETM
- [ ] Vergleich zwischen LDA und ETM auf dem Datensatz: 20NewsGroups (Maß: Topic Quality = Topic Coherence * Topic Diversity, Predictive Performance)
- [ ] Ergebnisse von ETM auf NYT-Stopwords und NYT-ohne-Stopwords
- [ ] Vergleich der Embedding zwischen CBOW und BERT-Wortembedding
- [ ] Vergleich zwischen prefitted-CBOW/SKIPGramm-ETM und prefitted-BERT-ETM

**Dokumentation für Teamarbeit**
- Einpaar Befehlen für die Nutzung von Google Colab für Teamarbeit:
1. !git config --global user.email "email@gmail.com"
2. !git config --global user.name "username"
3. !git clone https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space.git
4. cd /content/replication-topic-modelling-in-embedding-space
5. !git add /content/replication-topic-modelling-in-embedding-space
6. !git commit -m "test git push from google colab"
7. !git remote rm origin
8. !git remote add origin https://username:token@github.com/hanhluukim/replication-topic-modelling-in-embedding-space.git
9. !git push --set-upstream origin main
10. !git pull
