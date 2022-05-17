#!/usr/bin/env python
import os
import sys
import urllib
from urllib.parse import urlparse
try    : import json
except : import simplejson as json
from urllib.request import urlopen

#https://gist.github.com/christophermanning/1474925#file-bitbucket_download_directory-py


def open_directory_direct(directory_url):
    with urlopen(directory_url) as json_data_url_handle:
        if json_data_url_handle.code != 200:
            print(f'url {directory_url} not found')
            exit()
        json_directory = json.loads(json_data_url_handle.read())
        print(json_directory)

    
def open_directory(idx, API_PATH, username, repo_slug, path):
    #directory_url = "%s/%s/%s/%s" % (API_PATH, username, repo_slug, path)
    if idx==0:
        directory_url = "%s/%s/%s/src" % (API_PATH, username, repo_slug)
        print(directory_url)
        with urlopen(directory_url) as json_data_url_handle:
            if json_data_url_handle.code != 200:
                print(f'url {directory_url} not found')
                exit()
            json_directory = json.loads(json_data_url_handle.read())
            #print(json_directory['values'][0]['path'])
            #print(json_directory['values'][0]['links']['self']['href'])
            idx = idx + 1
            new_url = json_directory['values'][0]['links']['self']['href']
            open_directory_direct(new_url)
        
    else:
        directory_url = "%s/%s/%s/%s" % (API_PATH, username, repo_slug, path)
        print(directory_url)
        with urlopen(directory_url) as json_data_url_handle:
            if json_data_url_handle.code != 200:
                print(f'url {directory_url} not found')
                exit()
            json_directory = json.loads(json_data_url_handle.read())
            #print(json_directory['values'][0]['path'])
            #print(json_directory['values'][0]['links']['self']['href'])
            new_url = json_directory['values'][0]['links']['self']['href']
            #open_directory(new_url)
        
    #json_data_url_handle = urlopen(directory_url)
    #print(json_data_url_handle)

   
    
        
        """
        for directory in json_directory['directories']:
            open_directory(API_PATH, username, repo_slug, path + "/" + directory)

        for file in json_directory['files']:
            try:
                os.makedirs(os.path.dirname(file['path']))
            except OSError:
                None
            print(f'download: file["path"]')
            urllib.urlretrieve("%s/%s/%s/raw/%s/%s" % (API_PATH, username, repo_slug, file['revision'], file['path']), file['path'])
        """
"""
if (len(sys.argv) != 2 or sys.argv[1].find("https://bitbucket.org/") != 0 or sys.argv[1].find("/src/") == -1):
    print('usage: python bitbucket_download.py https://bitbucket.org/pypy/pypy/src/b590cf6de419/demo/')
    print('find the url by going to the source tab of a repository and browse to the directory you want to download')
    exit()
"""

given_url_1 = "https://bitbucket.org/franrruiz/data_nyt_largev_5/src/not_remove_glove/min_df_5000"
given_url_2 = "https://bitbucket.org/franrruiz/data_stopwords_largev_2/src/data_nyt/stopwords_not_remove_glove/min_df_5000"
API_PATH = "https://api.bitbucket.org/2.0/repositories"
null, username, repo_slug, path = urlparse(given_url_1).path.split("/", 3)
print(username)
print(repo_slug)
print(path)
open_directory(0, API_PATH, username, repo_slug, path)


null, username, repo_slug, path = urlparse(given_url_2)
open_directory(0, API_PATH, username, repo_slug, path)

"""
franrruiz
data_nyt_largev_5
src/not_remove_glove/min_df_5000
https://api.bitbucket.org/2.0/repositories/franrruiz/data_nyt_largev_5/src/647d7f60b845d2b271834dbc6d2d7fd3ada7d8d7/not_remove_glove/
"""


        

"""
def open_files_from_url(directory_url):
    print(f'starting reading files from {directory_url}')
    
    with urlopen(directory_url+"bow_tr_counts.mat") as json_data_url_handle:
        if json_data_url_handle.code != 200:
            print(f'url {directory_url} not found')
            exit()
        json_directory = json.loads(json_data_url_handle.read())
        print(json.dumps(json_directory, indent=4))
        for file in json_directory['files']:
            try:
                os.makedirs(os.path.dirname(file['path']))
            except OSError:
                None
            print(f'download: file["path"]')
            
    file_url = directory_url+"bow_tr_counts.mat"
    urlretrieve(file_url)
    
def open_directory(target_dir, directory_url):
    #print(target_directory_url)
    with urlopen(directory_url) as json_data_url_handle:
        if json_data_url_handle.code != 200:
            print(f'url {directory_url} not found')
            exit()
        json_directory = json.loads(json_data_url_handle.read())
        #print(json.dumps(json_directory, indent=4))
        target_dir_url = ""
        for i in range(len(json_directory['values'])):
            if target_dir in json_directory['values'][i]['links']['self']['href']:
                target_dir_url = json_directory['values'][i]['links']['self']['href']
                #print(new_url)
                open_files_from_url(target_dir_url)
    
def open_directory_first(API_PATH, repo_slug, target_dir):
    directory_url = "%s/%s/src" % (API_PATH, repo_slug)
    print(directory_url)
    with urlopen(directory_url) as json_data_url_handle:
        if json_data_url_handle.code != 200:
            print(f'url {directory_url} not found')
            exit()
        json_directory = json.loads(json_data_url_handle.read())
        #print(json.dumps(json_directory, indent=4))
        # get the src/id

        new_url = json_directory['values'][0]['links']['self']['href']
        print(f'start url: {new_url}')
        open_directory(target_dir, new_url)
        
            

#open_directory_first(API_PATH, repo_slug_1, target_dir_1)

"""
"""
franrruiz
data_nyt_largev_5
src/not_remove_glove/min_df_5000

main_url = https://api.bitbucket.org/2.0/repositories/franrruiz/data_nyt_largev_5/src/647d7f60b845d2b271834dbc6d2d7fd3ada7d8d7/
directory= not_remove_glove/
"""