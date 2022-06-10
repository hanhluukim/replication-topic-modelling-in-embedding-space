# replication-topic-modelling-in-embedding-space

**Ziel des Projekts**
1. Replizieren des Artikels: [Topic Modelling in embedding space](https://arxiv.org/abs/1907.04907) von Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei. Die [originale Implementierung](https://github.com/adjidieng/ETM) von Autoren. 
2. Neues Experiment: Kombination von pre-fitted BERT-Embeddings mit ETM
**Bericht**
1. Berichtschreiben im Overleaf: [Bericht](https://www.overleaf.com/read/wpfpwbxwwjhz), [HTML-Notebook-as-pdf](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/pdf_html_files/notebook_replication.pdf)

**Durchgeführte Experimenten**
- [x] Laufzeit des ETM Modells in der Abhängigkeit von der Vokabulargrößen
- [x] Vergleich zwischen LDA und ETM auf dem Datensatz: 20NewsGroups (Maß: Topic Quality = Topic Coherence * Topic Diversity, normalisierte-Perplexity)
- [x] Vergleich Ergebnisse von ETM auf 20NewsGroups mit Stopwords und ohne-Stopwords
- [x] Vergleich der Embedding zwischen Skipgramm und BERT-Wortembedding
- [x] Vergleich zwischen prefitted-Skipramm-ETM und prefitted-BERT-ETM

**Verwandte Pakette**
1. List aller verwendeten Paketten ist in der Datei: requirements.txt
2. Installieren durch: `pip install -r requirements.txt`
3. Wichtig sind: [gensim](https://radimrehurek.com/gensim/) für LDA, [torch](https://pytorch.org/docs/stable/torch.html) für variationale Inference, [sklearn](https://scikit-learn.org/stable/) für Datensatz
4. Anmerkung: Unsere Implementierung verwendet gensim=="3.8.3", sodass einige Parameters andere Namen hat als in dem neusten gensim (gensim=="4.2.0")

**Notwendige Dateien für BERT-prefitted-ETM**
1. Herunterladen die Bert-Embeddings from [Google-Drive-Link](https://drive.google.com/file/d/1aLLQCDFncdaedOS4pnB0-T6y0T7dHIck/view?usp=sharing): 
2. Packen die txt.Datei in dem Ordner: prepared_data/
3. Durchführen in dem Projektsordner: `python src/bert_covert_format.py`. Zwei Dateein: bert_embeddings.npy und bert_vocab.txt wurden automatisch in dem Ordner: `prepared_data/` gespeichert. 
4. Diese beiden Dateien sind notwendig für das Bert-prefitted-ETM

**Befehlen zum Durchführung von Experimenten**
1. LDA-Modelle: `python main_lda.py --filter-stopwords "True" --min-df 100 --epochs 20 --use-tensor True --batch-test-size 100` 
2. ETM-Modelle: `python main.py --model "ETM" --epochs 160 --wordvec-model "skipgram" --loss-name "cross-entropy" --min-df 2 --num-topics 20 --filter-stopwords "True" --hidden-size 800 --activate-func "ReLU" --optimizer-name "adam" --lr 0.002 --wdecay 0.0000012`
3. `--wordvec-model` kann folgende Werten haben: "cbow", "skipgram", "bert", 
4. `--optimizer-name` hat folgende Optionen "adam" und "sgd"

**Notebooks zum Durchführung von Experimenten**
1. [Notebook für LDA](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_topic_modelling_with_LDA.ipynb)
2. [Notebook für ETM](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_replication.ipynb)
3. [Notebook für Wordembeddings-BERT](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_bert_sentence_embeddings_to_word_embeddings.ipynb)
4. [Notebook für semantische Ähnlichkeiten](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_comparison_embeding_models.ipynb)

**Struktur des Repos**
1. Implementierung befindet sich im Ordner [src](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/tree/main/src)
2. Jeder Schritt von den durchgeführten Experimenten kann in der Datei [notebook_replication.ipynb](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_replication.ipynb) nachvollzieht und kontrolliert werden. Man kann dieses Notebook selbst auf dem Link Colab öffnen und durchführen. 
3. Jeder Schritt für Erstellung von Word-embeddings mit BERT in der Datei [notebook_bert_embedding.jpynb](https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space/blob/main/notebook_bert_sentence_embeddings_to_word_embeddings.ipynb)

**Genutzte Datensätze**
1. [20NewsGroups]()
2. Vorverarbeitungsschritten für Raw-Datensatz: Entfernen von nicht-alphabeltischen Zeichen, Stoppwörtern, Kleinschreiben, ...
3. Zerlegung: Traindatensatz (112114), Testdatensatz (Testdatensatz-h1, Testdatensatz-h2) (7532) und Validationssatz von 100 Dokumenten

**Angewandte Word-Embedding Methoden**
- [x] CBOW und Skipgram mittels [Gensim](https://radimrehurek.com/gensim/). Word-Embeddings werden für jedes Wort des Traindatensatzes gelernt. Es wird ein Trainingsprozess gebraucht, um Word-Repräsentationen zu haben
- [x] based-BERT-Wordembedding mittels [Transformer-Huggingface](https://huggingface.co/docs/transformers/installation). Benutzung von pretrained-BERT-Modell um direkt Repräsentationen für Wörter zu haben. Kein Trainieren ist benötig. 
- [x] Implementierung in : `src/embedding.py` für CBOW und Skipgram und für BERT: `src/bert_preparing.py`, `src/bert_embedding.py`, `bert_main.py`. Durchfühen Bert-Embedding mit `python bert_main.py` aus dem Hauptordner

**Architektur des ETM-Modells**
1. Einige wichtige Beschreibung:

- Das Vocabular V: besteht aus den einzigartigen Wörtern aus dem Traindatensatz
- Eingaben für ETM: (1) normalisierte-BOW für Encoder-Network und (2) Vocabular-Word-Embeddings
- Topic Embedding ist numersiche Repränstation für ein Topic in dem Embedding-Space, in dem Wort-Embeddings sich auch befinden

2. Das Modell ETM wird in `src/etm.py` implementiert.
