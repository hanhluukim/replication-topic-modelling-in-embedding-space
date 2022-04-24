# replication-topic-modelling-in-embedding-space

**Ziel**
1. Replizieren des Artikels: [Topic Modelling in embedding space](https://arxiv.org/abs/1907.04907) von Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei. Originales [Code](https://github.com/adjidieng/ETM) von Autoren. 
2. Neues Experiment: Kombination von pre-fitted BERT-Wordembedding mit ETM

**Datensatz**
1. 20NewsGroups
2. New York Times


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
