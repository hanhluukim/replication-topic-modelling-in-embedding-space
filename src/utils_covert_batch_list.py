def covert_to_list(theta_1, beta, batch_test_2):
    #theta of all documents
    theta_test_1_DxK = []
    for theta in theta_1:
        try:
            theta_test_1_DxK.append(theta.tolist())
        except:
            theta_test_1_DxK.append(list(theta))
    #beta: topics over words
    beta_KxV = []
    for bt in beta:
        try:
            beta_KxV.append(bt.tolist())
        except:
            beta_KxV.append(list(bt))
    count_of_each_word_in_doc_list_test_2 = []
    for doc in batch_test_2['bow']:
        count_words_in_doc = doc.tolist()
        count_of_each_word_in_doc_list_test_2.append(count_words_in_doc)
        
    return theta_test_1_DxK, beta_KxV, count_of_each_word_in_doc_list_test_2
