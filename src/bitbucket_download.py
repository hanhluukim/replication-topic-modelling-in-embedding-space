import argparse
import subprocess

parser = argparse.ArgumentParser(description='bitbucket download')
parser.add_argument('--username', type=str, default="LDA", help='bitbucket-username')
parser.add_argument('--app-pass', type=str, default="", help='bitbucket-app-pass')

args = parser.parse_args()
      

host_username = "franrruiz"
API_PATH = f'https://api.bitbucket.org/2.0/repositories/{host_username}'


repo_slug_1 = "data_nyt_largev_5"
target_dir_1 = "min_df_5000"

repo_slug_2 = "data_stopwords_largev_2"
target_dir_2 = "min_df_5000"

file_names = [			
"bow_tr_counts.mat",
"bow_tr_tokens.mat",
"bow_ts_counts.mat",
"bow_ts_h1_counts.mat",
"bow_ts_h1_tokens.mat",
"bow_ts_h2_counts.mat",
"bow_ts_h2_tokens.mat",
"bow_ts_tokens.mat",
"bow_va_counts.mat",
"bow_va_tokens.mat",
"vocab.pkl",   
]

file_names_new = [
    "bow_tr_tokens.mat",
]

dir1= ("without-stopwords", "not_remove_glove/min_df_5000")
dir2= ("with-stopwords", "data_nyt/stopwords_not_remove_glove/min_df_5000")


repo_list = {repo_slug_1:dir1, repo_slug_2:dir2}
#repo_list_new = {repo_slug_1:dir1}

for repo_slug, folder_name in repo_list.items():
    for file_name in file_names:
        download_url = f'https://api.bitbucket.org/2.0/repositories/{host_username}/{repo_slug}/src/master/{folder_name[1]}/{file_name}'
        print(download_url)
        out_file = f'./prepared_data/new_york_time_data/{folder_name[0]}/{file_name}'
        print(out_file)
        user_info = f'{args.username}:{args.app_pass}'

        subprocess.run(
            ["curl", 
             "-u", user_info,
             download_url,
             "-o", out_file]
            )      
