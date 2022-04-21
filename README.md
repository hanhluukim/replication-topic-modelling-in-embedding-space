# replication-topic-modelling-in-embedding-space
- Einpaar Befehlen für die Nutzung von Google Colab:
!git config --global user.email ""
!git config --global user.name ""
!git clone https://github.com/hanhluukim/replication-topic-modelling-in-embedding-space.git
cd /content/replication-topic-modelling-in-embedding-space
!git add /content/replication-topic-modelling-in-embedding-space
!git commit -m "test git push from google colab"
!git remote rm origin
!git remote add origin https://<username>:<token>@github.com/<username>/replication-topic-modelling-in-embedding-space.git
!git push --set-upstream origin main
!git pull
