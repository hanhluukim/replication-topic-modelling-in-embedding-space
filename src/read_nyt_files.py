#using min_df 5000 without stopwords
#using min_df 5000 with stopwords
#nead documents in words

#https://bitbucket.org/franrruiz/data_stopwords_largev_2/src/master/data_nyt/stopwords_not_remove_glove/min_df_5000/
#https://bitbucket.org/franrruiz/data_nyt_largev_5/src/master/not_remove_glove/min_df_5000/

            
import scipy.io
def read_mat_file(file_name):   
    mat = scipy.io.loadmat(file_name)
    return mat

fp = "prepared_data/new_york_time_data/without-stopwords/bow_tr_counts.mat"
print(read_mat_file(fp))

def cover_to_bows(set_name, set_in_mat):
    tokens = set_in_mat['tokens']
    counts = set_in_mat['counts']
    n_docs_in_set = len(set_in_mat['tokens'])

