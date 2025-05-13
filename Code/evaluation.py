from util import *
from collections import defaultdict

# Add your import statements here

class Evaluation():
    def group_qrels(self, qrels):
        # qrels_grouped = {}
        qrels_grouped = defaultdict(list)
        for entry in qrels:
            query_num = int(entry['query_num'])
            # print(type(query_num))
            doc_id = int(entry['id'])
            qrels_grouped[query_num].append(doc_id)
        return qrels_grouped

    def group_qrels1(self, qrels):
        qrels_grouped = defaultdict(list)
        for entry in qrels:
            query_num = int(entry['query_num'])
            doc_id = int(entry['id'])
            position = int(entry['position'])  # <-- read position
            qrels_grouped[query_num].append((doc_id, position))
        return qrels_grouped



    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        precision = -1

        #Fill in code here
        retrieved_docs = query_doc_IDs_ordered[:k]
        # print(retrieved_docs)
        relevant_retrieved = [doc_id for doc_id in retrieved_docs if doc_id in true_doc_IDs]
        precision = len(relevant_retrieved) / k
        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        meanPrecision = -1
        # print('doc_IDs_ordered',len(doc_IDs_ordered)) # 225
        # for i in doc_IDs_ordered:
        #     print(len(i))  # 1400
        #     print(i[0])
        # print('query_ids', query_ids)  # query_ids [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225]
        # print('qrels', qrels)
        
        #Fill in code here
        # First, group qrels by query_id
        qrels_grouped = self.group_qrels(qrels)
        precision_list = []
        # print(qrels_grouped)  #  {1: [184, 29, 31, 12, 51, 102, 13, 14, 15, 57, 378, 859, 185, 30, 37, 52, 142, 195, 875, 56, 66, 95, 462, 497, 858, 876, 879, 880, 486], 2: [12, 15, 184, 858, 51, 102, 202, 14, 52, 380, 746, 859, 948, 285, 390, 391, 442, 497, 643, 856, 857, 877, 864, 658, 486], ..., }
        # print(qrels_grouped.keys()) # dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225])
        for i, query_id in enumerate(query_ids):
            #true_docs = qrels_grouped.get(query_id, {'relevant_docs': []})['relevant_docs']
            # print(true_docs)
            true_docs = qrels_grouped[query_id]
            precision = self.queryPrecision(doc_IDs_ordered[i], query_id, true_docs, k)
            precision_list.append(precision)
            # print(len(precision_list)) # for each k from 1 to 10: precision over 225 queries: for each query get precision based on k retrieved docs
        
        return np.mean(precision_list)
    
    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        recall = -1

        #Fill in code here
        retrieved_docs = query_doc_IDs_ordered[:k]
        relevant_retrieved = [doc_id for doc_id in retrieved_docs if doc_id in true_doc_IDs]
        recall = len(relevant_retrieved) / len(true_doc_IDs) if len(true_doc_IDs) > 0 else 0
        return recall


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        meanRecall = -1

        #Fill in code here
        # Group qrels by query_id
        qrels_grouped = self.group_qrels(qrels)
        recall_list = []
        
        for i, query_id in enumerate(query_ids):
            #true_docs = qrels_grouped.get(query_id, {'relevant_docs': []})['relevant_docs']
            true_docs = qrels_grouped[query_id]
            recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_docs, k)
            recall_list.append(recall)
        
        return np.mean(recall_list)

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        fscore = -1

        #Fill in code here
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        beta = 0.5
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value
        
        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1

        #Fill in code here
        qrels_grouped = self.group_qrels(qrels)
        fscore_list = []
        
        for i, query_id in enumerate(query_ids):
            #true_docs = qrels_grouped.get(query_id, {'relevant_docs': []})['relevant_docs']
            true_docs = qrels_grouped[query_id]
            fscore = self.queryFscore(doc_IDs_ordered[i], query_id, true_docs, k)
            fscore_list.append(fscore)
        
        return np.mean(fscore_list)

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        # nDCG = -1
        # #Fill in code here
        # retrieved_docs = query_doc_IDs_ordered[:k]
        # DCG = 0
        # print(retrieved_docs)
        # print(true_doc_IDs)
        # for i, doc_id in enumerate(retrieved_docs):
        #     if doc_id in true_doc_IDs:
        #         DCG += 1 / math.log2(i + 2)  # position i corresponds to rank i+1
        # # Ideal DCG
        # ideal_relevant = min(len(true_doc_IDs), k)
        # IDCG = sum(1 / math.log2(i + 2) for i in range(ideal_relevant))
        # nDCG = DCG / IDCG if IDCG != 0 else 0
        # return nDCG
        nDCG = -1
        retrieved_docs = query_doc_IDs_ordered[:k]
        DCG = 0
    
        # Build a dict for fast lookup: doc_id -> position (relevance score)
        true_relevance_dict = dict(true_doc_IDs)
        # print('true_relevance_dict',true_relevance_dict)
        # print('retrieved_docs',retrieved_docs)
        rel_ideal = []
        for i, doc_id in enumerate(retrieved_docs):
            rel = true_relevance_dict.get(doc_id, 5)  # 0 if not found
            # print('rel', rel)
            # DCG += (2**rel - 1) / math.log2(i + 2)
            # if rel != 0:
            DCG += (5 - rel) / math.log2(i + 2) 
            rel_ideal.append(5-rel)
            # else:
        # Now calculate IDCG (ideal DCG)
        # Sort the true relevances from highest to lowest
        # ideal_rels = sorted([rel for (doc_id, rel) in true_doc_IDs], reverse=False)
        rel_ideal.sort(reverse = True)
        # print(rel_ideal)
        # print(ideal_rels)
        # ideal_rels = ideal_rels[:k]  # Only consider top-k
        # print(ideal_rels)
        # IDCG = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
        # for rel in ieal_rels:
            # print('idealrel', rel)
        IDCG = sum((rel / math.log2(i + 2) for i, rel in enumerate(rel_ideal)))
    
        nDCG = DCG / IDCG if IDCG != 0 else 0
        return nDCG

        # def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs_with_relevance, k):



    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        # meanNDCG = -1

        # #Fill in code here
        # print(qrels)
        # qrels_grouped = self.group_qrels(qrels)
        # nDCG_list = []
        
        # for i, query_id in enumerate(query_ids):
        #     #true_docs = qrels_grouped.get(query_id, {'relevant_docs': []})['relevant_docs']
        #     true_docs = qrels_grouped[query_id]
        #     nDCG = self.queryNDCG(doc_IDs_ordered[i], query_id, true_docs, k)
        #     nDCG_list.append(nDCG)
        
        # return np.mean(nDCG_list)
        qrels_grouped = self.group_qrels1(qrels)
        nDCG_list = []
        
        for i, query_id in enumerate(query_ids):
            true_docs_with_relevance = qrels_grouped[query_id]
            # print('true_docs_with_relevance',true_docs_with_relevance)
            nDCG = self.queryNDCG(doc_IDs_ordered[i], query_id, true_docs_with_relevance, k)
            nDCG_list.append(nDCG)
        
        return np.mean(nDCG_list)

    # def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        avgPrecision = -1

        #Fill in code here
        retrieved_docs = query_doc_IDs_ordered[:k]
        num_relevant = 0
        precision_sum = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in true_doc_IDs:
                num_relevant += 1
                precision = num_relevant / (i + 1)
                precision_sum += precision
        avgPrecision = precision_sum / len(true_doc_IDs) if len(true_doc_IDs) > 0 else 0
        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        #Fill in code here
        qrels_grouped = self.group_qrels(qrels)
        AP_list = []
        
        for i, query_id in enumerate(query_ids):
            #true_docs = qrels_grouped.get(query_id, {'relevant_docs': []})['relevant_docs']
            true_docs = qrels_grouped[query_id]
            AP = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_docs, k)
            AP_list.append(AP)
        
        return np.mean(AP_list)