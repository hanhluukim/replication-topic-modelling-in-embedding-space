# replication-topic-modelling-in-embedding-space

**Ziel des Projekts**
1. Replizieren des Artikels: [Topic Modelling in embedding space](https://arxiv.org/abs/1907.04907) von Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei. Die [originale Implementierung](https://github.com/adjidieng/ETM) von Autoren. 
2. Neues Experiment: Kombination von pre-fitted BERT-Embedding mit ETM
3. (optional) Vergleich der Laufzeit beim Trainieren zwischen dem python-ETM und dem julia-ETM

**Verwandte Pakette**
1. List aller verwendeten Paketten ist in der Datei: requirements.txt
2. Installieren durch: `pip install -r requirements.txt`

**Durchführung von Experimenten**
1. LDA-Modelle: `python main_lda.py --filter-stopwords "True" --min-df 100 --epochs 5 --use-tensor True --batch-test-size 100` 
2. ETM-Modelle: `python main.py --model "ETM" --epochs 160 --wordvec-model "bert" --loss-name "cross-entropy" --min-df 2 --num-topics 20 --filter-stopwords "True" --hidden-size 800 --activate-func "ReLU" --optimizer-name "adam" --lr 0.003 --wdecay 0.0000012`
3. `--wordvec-model` kann folgende Werten haben: "cbow", "skipgram", "bert", `--optimizer-name` hat folgende Optionen "adam" und "sgd"

**Struktur des Repos**
1. Implementierung befindet sich im Ordner [src](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/tree/main/src)
2. Jeder Schritt von den durchgeführten Experimenten kann in der Datei [notebook_replication.ipynb](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_replication.ipynb) nachvollzieht und kontrolliert werden. Man kann dieses Notebook selbst auf dem Link Colab öffnen und durchführen. 
3. Jeder Schritt für Erstellung von Word-embeddings mit BERT in der Datei [notebook_bert_embedding.jpynb](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_bert_embedding.ipynb)


**Genutzte Datensätze**
1. [20NewsGroups]()
2. Vorverarbeitungsschritten für Raw-Datensatz: Entfernen von nicht-alphabeltischen Zeichen, Stoppwörtern, Kleinschreiben, ...
3. Zerlegung: Traindatensatz, Testdatensatz (Testdatensatz-h1, Testdatensatz-h2) und Validationssatz von 100 Dokumenten

**Angewandte Word-Embedding Methoden**
- [ ] CBOW und Skipgram mittels [Gensim](https://radimrehurek.com/gensim/). Word-Embeddings werden für jedes Wort des Traindatensatzes gelernt. Es wird ein Trainingsprozess gebraucht, um Word-Repräsentationen zu haben
- [ ] based-BERT-Wordembedding mittels [Transformer-Huggingface](https://huggingface.co/docs/transformers/installation). Benutzung von pretrained-BERT-Modell um direkt Repräsentationen für Wörter zu haben. Kein Trainieren ist benötig. 
- [ ] Implementierung in : `src/embedding.py` für CBOW und Skipgram und für BERT: `src/bert_preparing.py`, `src/bert_embedding.py`, `bert_main.py`. Durchfühen Bert-Embedding mit `python bert_main.py` aus dem Hauptordner

**Architektur des ETM-Modells**
1. Einige wichtige Beschreibung:

- Das Vocabular V: besteht aus den einzigartigen Wörtern aus dem Traindatensatz
- Eingaben für ETM: (1) normalisierte-BOW für Encoder-Network und (2) Vocabular-Word-Embeddings
- Topic Embedding ist numersiche Repränstation für ein Topic in dem Embedding-Space, in dem Wort-Embeddings sich auch befinden

2. Das Modell ETM wird in `src/etm.py` implementiert.

**Durchgeführte Experimenten**

- [ ] Vergleich Top-5-Wörter von den top 7 meisten genutzen Topics (Datensatz: 1.8M Documents von NYT, Corpus V=212237, K=300) zwischen LDA und ETM
- [ ] Vergleich zwischen LDA und ETM auf dem Datensatz: 20NewsGroups (Maß: Topic Quality = Topic Coherence * Topic Diversity, normalisierte-Perplexity)
- [ ] Vergleich Ergebnisse von 20NewsGroup auf Stopwords und ohne-Stopwords
- [ ] Vergleich der Embedding zwischen Skipgramm und BERT-Wortembedding
- [ ] Vergleich zwischen prefitted-Skipramm-ETM und prefitted-BERT-ETM

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

**Unklarheiten aus dem originalen Paper und Implementierung**
1. Keine festlegene Zerlegung von Datensätzen (Trainset kann unterschiedlich bei den unterschiedlichen Durchführungen sein)
2. Keine klare Information über Trainingsparameters wie Learn-rate, ...
3. Keine klare Information über das Netzarchitektur wie Hidden-Layer-Dimensionen, ...
