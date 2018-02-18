import scipy.sparse as sp
import numpy as np
import os.path
from PIL import Image
import tensorflow as tf
from collections import defaultdict
import random
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm
import csv
from gensim.models import Word2Vec
import string
class Dataset(object):

    def __init__(self, rating_matrix_path,num_negatives,batch_size,data_set='citeulike-a',text_path=None,word2vec_model_path=None):
        pass
        self.epoch=0
        self.batch_index=-1
        self.num_negatives=num_negatives
        self.batch_size=batch_size
        #choose citeulike or ml-1m
        #citeulike data
        self.trainMatrix,self.validationMatrix,self.testMatrix,self.user_num,self.item_num,self.new_item_index=self.read_citeulike_rating_matrix(rating_matrix_path)
        self.test_item_id,self.test_labels=self.get_BPR_test_instances()
        self.train_users,self.train_items,self.batch_num,self.train_num=self.get_train_user_item(self.trainMatrix)
        self.users,self.pos_items,self.neg_item_set,self.neg_item_index=self.get_train_instances()
        if data_set=='citeulike-a':
            self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec=self.read_citeulike_a_text(text_path,word2vec_model_path)
        elif data_set=='citeulike-t':
            self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec=self.read_citeulike_t_text(text_path,word2vec_model_path)
        elif data_set=='movielens-1m' or data_set=='movielens-10m':
            self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec=self.read_movielens_1m(text_path,word2vec_model_path)
        else:
            print 'no text data'

    def next_batch(self):
        self.batch_index+=1
        if self.batch_index>=self.batch_num:
            self.batch_index=0
            self.epoch+=1
            self.users,self.pos_items,self.neg_item_set,self.neg_item_index=self.get_train_instances()
        return self.users[self.batch_index],self.pos_items[self.batch_index],self.neg_item_set[self.batch_index],self.neg_item_index[self.batch_index]
    
    # def next_BPR_batch(self,batch_size):
    #     start=self.index_in_epoch
    #     self.index_in_epoch+=batch_size
    #     if self.index_in_epoch>self.train_num:
    #         self.epoch+=1
    #         self.items,self.positive_users,self.negative_users,_=self.get_train_instances(self.trainMatrix)
    #         start=0
    #         self.index_in_epoch=batch_size
    #     end=self.index_in_epoch

    #     return self.items[start:end],self.positive_users[start:end],self.negative_users[start:end]

    def get_doc(self):
        return self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec

    #get test ids and labels
    def get_BPR_test_instances(self):
        test_labels=[]
        test_item_id=[]
        for i in xrange(self.user_num):
            one_user_item_id=[]
            one_user_labels=[]
            for j in xrange(self.item_num):
                if self.trainMatrix.has_key((i, j)) or self.validationMatrix.has_key((i, j)):
                    continue
                if self.testMatrix.has_key((i, j)):
                    one_user_labels.append(1)
                else:
                    one_user_labels.append(0)
                one_user_item_id.append(j)
            one_user_item_id=np.array(one_user_item_id)
            one_user_labels=np.array(one_user_labels)
            test_item_id.append(one_user_item_id)
            test_labels.append(one_user_labels)
        return test_item_id,test_labels

    #get all test data for one user
    #BPR_test:[test_num]
    #labels:[test_num]
    def get_BPR_test(self,user_id):
        # BPR_test=[]
        # labels=[]
        # for i in xrange(self.item_num):
        #     if self.trainMatrix.has_key((user_id, i)) or self.validationMatrix.has_key((user_id, i)):
        #         continue
        #     if self.testMatrix.has_key((user_id, i)):
        #         labels.append(1)
        #     else:
        #         labels.append(0)
        #     BPR_test.append(i)
        # BPR_test=np.array(BPR_test)
        # labels=np.array(labels)
        item_id=self.test_item_id[user_id]
        labels=self.test_labels[user_id]
        return item_id,labels

    #users:[instances]
    #items:[instances]
    #batch_num
    def get_train_user_item(self,train):
        users,items=[],[]
        for (u,i) in train.keys():
            users.append(u)
            items.append(i)
        users=np.array(users)
        items=np.array(items)
        batch_num=int(users.shape[0]/self.batch_size)
        train_num=users.shape[0]
        return users,items,batch_num,train_num

    #users:[batch_num,batch_size]
    #pos_items:[batch_num,batch_size]
    #neg_item_set:[batch_num,batch_size] a set of negative items for a batch
    #neg_item_index:[batch_num,batch_size,num_neg]
    def get_train_instances(self):
        #shuffle data
        perm=np.arange(self.train_num)
        np.random.shuffle(perm)
        self.train_users=self.train_users[perm]
        self.train_items=self.train_items[perm]

        users=np.reshape(self.train_users[0:self.batch_num*self.batch_size],(self.batch_num,self.batch_size))
        pos_items=np.reshape(self.train_items[0:self.batch_num*self.batch_size],(self.batch_num,self.batch_size))
        neg_item_set=np.random.randint(low=0,high=self.item_num,size=(self.batch_num,self.batch_size),dtype=np.int32)
        neg_item_index=np.zeros([self.batch_num,self.batch_size,self.num_negatives],dtype=np.int32)
        for i in xrange(self.batch_num):
            for j in xrange(self.batch_size):
                one_neg_group=[]
                for k in xrange(self.num_negatives):
                    #user_id:users[i,j]
                    item_index=np.random.randint(self.batch_size)
                    item_id=neg_item_set[i,item_index]
                    while self.trainMatrix.has_key((users[i,j], item_id)) or item_index in one_neg_group:
                        item_index=np.random.randint(self.batch_size)
                        item_id=neg_item_set[i,item_index]
                    one_neg_group.append(item_index)
                one_neg_group=np.array(one_neg_group)
                neg_item_index[i,j,:]=one_neg_group
        return users,pos_items,neg_item_set,neg_item_index


    # #items:[instances]
    # #positive_users:[instances]
    # #negative_users:[instances]
    # def get_train_instances(self,train):
    #     items,positive_users,negative_users=[],[],[]
    #     for (u,i) in train.keys():
    #         items.append(i)
    #         positive_users.append(u)
    #         one_negative_user_group=[]
    #         for num in xrange(self.num_negatives):
    #             neg_user=np.random.randint(self.user_num)
    #             while self.trainMatrix.has_key((neg_user, i)) or neg_user in one_negative_user_group:
    #                 neg_user=np.random.randint(self.user_num)
    #             one_negative_user_group.append(neg_user)
    #         one_negative_user_group=np.array(one_negative_user_group)
    #         negative_users.append(one_negative_user_group)
    #     items=np.array(items)
    #     positive_users=np.array(positive_users)
    #     negative_users=np.array(negative_users)
    #     train_num=items.shape[0]
    #     #shuffle
    #     perm=np.arange(train_num)
    #     np.random.shuffle(perm)
    #     items=items[perm]
    #     positive_users=positive_users[perm]
    #     negative_users=negative_users[perm]
    #     return items,positive_users,negative_users,train_num

    #using CML method
    #BPR_user_input:[instances]
    #BPR_positive_item:[instances]
    #BPR_negative_item:[instances,num_neg]
    # def get_BPR_train_instances(self,train):
    #     BPR_user_input,BPR_positive_item,BPR_negative_item=[],[],[]
    #     for (u,i) in train.keys():
    #         # positive instances
    #         BPR_user_input.append(u)
    #         BPR_positive_item.append(i)
    #         one_negative_item_group=[]
    #         for num in xrange(self.num_negatives):
    #             #negative instances
    #             j = np.random.randint(self.item_num)
    #             while self.trainMatrix.has_key((u, j)) or j in one_negative_item_group:
    #                 j = np.random.randint(self.item_num)
    #             one_negative_item_group.append(j)
    #         one_negative_item_group=np.array(one_negative_item_group)
    #         BPR_negative_item.append(one_negative_item_group)
    #     BPR_user_input=np.array(BPR_user_input)
    #     BPR_positive_item=np.array(BPR_positive_item)
    #     BPR_negative_item=np.array(BPR_negative_item)
    #     train_num=BPR_user_input.shape[0]
    #     #shuffle
    #     perm=np.arange(train_num)
    #     np.random.shuffle(perm)
    #     BPR_user_input=BPR_user_input[perm]
    #     BPR_positive_item=BPR_positive_item[perm]
    #     BPR_negative_item=BPR_negative_item[perm]
    #     return BPR_user_input,BPR_positive_item,BPR_negative_item,train_num

    def get_user_item_num(self):
        return self.user_num,self.item_num

    def get_train_num(self):
        return self.train_num

    def get_epoch(self):
        return self.epoch

    #split data into train,validation,test
    def split_data(self,user_item_matrix, split_ratio=(4, 0, 1), seed=1):
        np.random.seed(seed)
        train = dok_matrix(user_item_matrix.shape)
        validation = dok_matrix(user_item_matrix.shape)
        test = dok_matrix(user_item_matrix.shape)
        user_item_matrix = lil_matrix(user_item_matrix)
        for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
            items = list(user_item_matrix[user, :].nonzero()[1])
            if len(items) >= 5:

                np.random.shuffle(items)

                train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
                valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

                for i in items[0: train_count]:
                    train[user, i] = 1
                for i in items[train_count: train_count + valid_count]:
                    validation[user, i] = 1
                for i in items[train_count + valid_count:]:
                    test[user, i] = 1
        print("{}/{}/{} train/valid/test samples".format(
            len(train.nonzero()[0]),
            len(validation.nonzero()[0]),
            len(test.nonzero()[0])))
        return train, validation, test

    def read_citeulike_rating_matrix(self,path):
        user_dict = defaultdict(set)
        for u, item_list in enumerate(open(path).readlines()):
            items = item_list.strip().split(" ")
            for item in items:
                user_dict[u].add(int(item))

        n_users = len(user_dict)
        n_items = max([item for items in user_dict.values() for item in items]) + 1

        user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
        for u, item_list in enumerate(open(path).readlines()):
            items = item_list.strip().split(" ")
            for item in items:
                user_item_matrix[u,int(item)]=1

        #get tag
        # n_features = 0
        # for l in open(tag_path).readlines():
        #     items = l.strip().split(" ")
        #     if len(items) >= tag_occurence_thres:
        #         n_features += 1
        # print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))
        # features = dok_matrix((n_items, n_features), dtype=np.int32)
        # feature_index = 0
        # for l in open(tag_path).readlines():
        #     items = l.strip().split(" ")
        #     if len(items) >= tag_occurence_thres:
        #         features[[int(i) for i in items], feature_index] = 1
        #         feature_index += 1

        #new_item_index,user_item_matrix=self.sort_item(user_item_matrix,n_users,n_items)
        train,validation,test=self.split_data(user_item_matrix)
        new_item_index=0
        

        return train,validation,test,n_users,n_items,new_item_index

    #sort the items according to their frequency
    def sort_item(self,matrix,user_num,item_num):
        item_count=np.zeros([item_num],dtype=np.int32)
        for (u,i) in matrix.keys():
            item_count[i]+=1

        new_item_index=np.argsort(-item_count)
        new_item_index_transpose=np.zeros([item_num],dtype=np.int32)
        for i in xrange(item_num):
            new_item_index_transpose[new_item_index[i]]=i
        user_item_matrix = dok_matrix((user_num, item_num), dtype=np.int32)
        for (u,i) in matrix.keys():
            new_i=new_item_index_transpose[i]
            user_item_matrix[u,new_i]=1
        # new_features=dok_matrix((item_num,features_num),dtype=np.int32)
        # for (i,f) in features.keys():
        #     new_i=new_item_index_transpose[i]
        #     new_features[new_i,f]=1
        return new_item_index,user_item_matrix

    #sort the words according to their frequency
    def sort_word(self,doc):
        pass
        dic_count={}
        for paper in doc:
            for sent in paper:
                for word in sent:
                    if(dic_count.get(word,-1)==-1):
                        dic_count[word]=1
        word_num=len(dic_count)

        dic_index={}
        index=0
        frequency=np.zeros(word_num,dtype=np.int32)
        for paper in doc:
            for sent in paper:
                for word in sent:
                    if(dic_index.get(word,-1)==-1):
                        
                        dic_index[word]=index
                        frequency[index]+=1
                        index+=1
                    else:
                        frequency[dic_index[word]]+=1
        new_word_index=np.argsort(-frequency)
        new_word_index_transpose=np.zeros([word_num],dtype=np.int32)
        for i in xrange(word_num):
            new_word_index_transpose[new_word_index[i]]=i
        dic_sorted={}
        for paper in doc:
            for sent in paper:
                for word in sent:
                    index=dic_index[word]
                    new_index=new_word_index_transpose[index]
                    dic_sorted[word]=new_index
        return dic_sorted


    #return word2vec of text
    #doc_array:[item_num,max_sent,max_words_per_sent]
    #mask:[item_num,max_sent]  save the number of words per sentence
    #mask_bool:[item_num,max_sent,max_words_per_sent]  1 for valid word,0 for invalid word
    #word_vec:[words] vector of words
    def read_movielens_1m(self,text_path,model_path):
        doc=[]
        punctuation='!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
        nextline='\n'

        with open(text_path, "r") as f:
            paper=f.readline()
            while paper != None and paper != "":
                paper=paper.split('::')
                paper=paper[2]
                paper=paper.lower()
                for i in punctuation:
                    paper=paper.replace(i,' ')
                for i in nextline:
                    paper=paper.replace(i,' ')
                #print line
                #split sentences
                line=[]
                sentences=paper.split('.')
                for one_sen in sentences:
                    array=one_sen.split(' ')
                    while '' in array:
                        array.remove('')
                    line.append(array)
                doc.append(line)
                paper=f.readline()
        print 'read doc complete'
        #remove common words and words that only appear once
        frequency = defaultdict(int)
        stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
        for line in doc:
            for sent in line:
                for word in sent:
                    frequency[word]+=1
        doc=[[[word for word in sent if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
                for sent in line]for line in doc]  
        #print doc
        #remove sentences that are too short
        for i in xrange(len(doc)):
            for j in xrange(len(doc[i])-1,-1,-1):
                if len(doc[i][j])<2:
                    doc[i].pop(j)
        #print doc

        #cut the line if it is too long
        max_sent=15
        max_words_per_sent=15
        for i in xrange(len(doc)):
            if len(doc[i])>max_sent:
                doc[i]=doc[i][0:max_sent]
            for j in xrange(len(doc[i])):
                if len(doc[i][j])>max_words_per_sent:
                    doc[i][j]=doc[i][j][0:max_words_per_sent]
        print 'preprocess doc complete'

        dic=self.sort_word(doc)#dictionary from word to index
        #print doc
        #add word dictionary. doc will only save indics
        #dic={} #dictionary from word to index
        doc_index=[]#same size as doc. but it save indics instead of word
        for paper in doc:
            paper_index=[]
            for sent in paper:
                sent_index=[]
                for word in sent:
                    sent_index.append(dic[word])
                paper_index.append(sent_index)
            doc_index.append(paper_index)

        #convert list into array
        #get mask for each sent
        doc_array=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.int32)
        doc_array_reverse=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.int32)
        mask=np.zeros([len(doc_index),max_sent],dtype=np.int32)
        mask_bool=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.float32)
        for i in xrange(len(doc_index)):
            for j in xrange(len(doc_index[i])):
                mask[i][j]=len(doc_index[i][j])
                for k in xrange(len(doc_index[i][j])):
                    doc_array[i][j][k]=doc_index[i][j][k]
                    doc_array_reverse[i][j][mask[i][j]-1-k]=doc_index[i][j][k]
                    mask_bool[i][j][k]=1

        print 'get mask complete'

        #get word vectors
        model=Word2Vec.load(model_path)
        word_vec=np.zeros([len(dic)+1,model['are'].shape[0]],dtype=np.float32) #size:word_num*vector_per_word
        for word in dic:
            vec=model[word]
            word_vec[dic[word],:]=vec
        #print dic
        #print doc_index
        #print doc_array
        #print mask
        # doc_array=doc_array[self.new_item_index]
        # doc_array_reverse=doc_array_reverse[self.new_item_index]
        # mask=mask[self.new_item_index]
        # mask_bool=mask_bool[self.new_item_index]

        print 'read text data complete.'
        return doc_array,doc_array_reverse,mask,mask_bool,word_vec

    #return word2vec of text
    #doc_array:[item_num,max_sent,max_words_per_sent]
    #mask:[item_num,max_sent]  save the number of words per sentence
    #mask_bool:[item_num,max_sent,max_words_per_sent]  1 for valid word,0 for invalid word
    #word_vec:[words] vector of words
    def read_citeulike_t_text(self,text_path,model_path):
        doc=[]
        punctuation='!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
        nextline='\n'
        with open(text_path, "r") as f:
            paper = f.readline()
            #one line text, one line number
            text=False
            while paper != None and paper != "":
                if text:
                    text=False
                    paper=paper.lower()
                    for i in punctuation:
                        paper=paper.replace(i,' ')
                    for i in nextline:
                        paper=paper.replace(i,' ')
                    #print line
                    #split sentences
                    line=[]
                    sentences=paper.split('.')
                    for one_sen in sentences:
                        array=one_sen.split(' ')
                        while '' in array:
                            array.remove('')
                        line.append(array)
                    doc.append(line)
                    paper=f.readline()
                else:
                    text=True
                    paper=f.readline()
        
        #remove common words and words that only appear once
        frequency = defaultdict(int)
        stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
        for line in doc:
            for sent in line:
                for word in sent:
                    frequency[word]+=1
        doc=[[[word for word in sent if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
                for sent in line]for line in doc]  
        #print doc
        #remove sentences that are too short
        for i in xrange(len(doc)):
            for j in xrange(len(doc[i])-1,-1,-1):
                if len(doc[i][j])<2:
                    doc[i].pop(j)
        #print doc

        #cut the line if it is too long
        max_sent=15
        max_words_per_sent=15
        for i in xrange(len(doc)):
            if len(doc[i])>max_sent:
                doc[i]=doc[i][0:max_sent]
            for j in xrange(len(doc[i])):
                if len(doc[i][j])>max_words_per_sent:
                    doc[i][j]=doc[i][j][0:max_words_per_sent]

        dic=self.sort_word(doc)#dictionary from word to index
        #print doc
        #add word dictionary. doc will only save indics
        #dic={} #dictionary from word to index
        doc_index=[]#same size as doc. but it save indics instead of word
        for paper in doc:
            paper_index=[]
            for sent in paper:
                sent_index=[]
                for word in sent:
                    sent_index.append(dic[word])
                paper_index.append(sent_index)
            doc_index.append(paper_index)

        #convert list into array
        #get mask for each sent
        doc_array=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.int32)
        doc_array_reverse=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.int32)
        mask=np.zeros([len(doc_index),max_sent],dtype=np.int32)
        mask_bool=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.float32)
        for i in xrange(len(doc_index)):
            for j in xrange(len(doc_index[i])):
                mask[i][j]=len(doc_index[i][j])
                for k in xrange(len(doc_index[i][j])):
                    doc_array[i][j][k]=doc_index[i][j][k]
                    doc_array_reverse[i][j][mask[i][j]-1-k]=doc_index[i][j][k]
                    mask_bool[i][j][k]=1
        #get word vectors
        model=Word2Vec.load(model_path)
        word_vec=np.zeros([len(dic)+1,model['are'].shape[0]],dtype=np.float32) #size:word_num*vector_per_word
        for word in dic:
            vec=model[word]
            word_vec[dic[word],:]=vec
        #print dic
        #print doc_index
        #print doc_array
        #print mask
        # doc_array=doc_array[self.new_item_index]
        # doc_array_reverse=doc_array_reverse[self.new_item_index]
        # mask=mask[self.new_item_index]
        # mask_bool=mask_bool[self.new_item_index]
        
        print 'read text data complete.'
        return doc_array,doc_array_reverse,mask,mask_bool,word_vec



    #return word2vec of text
    #doc_array:[item_num,max_sent,max_words_per_sent]
    #doc_array_reverse:[item_num,max_sent,max_words_per_sent]
    #mask:[item_num,max_sent]  save the number of words per sentence
    #mask_bool:[item_num,max_sent,max_words_per_sent]  1 for valid word,0 for invalid word
    #word_vec:[words] vector of words
    def read_citeulike_a_text(self,text_path,model_path):
        doc=[]
        with open(text_path,'r') as f:
            reader=csv.reader(f)
            first=1
            for row in reader:
                if first>0:
                    first=0
                    continue
                paper=row[1].lower()+'.'+row[4].lower()
                punctuation='!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
                #remove punctuation marks
                for i in '{}':
                    paper=paper.replace(i,'')
                for i in punctuation:
                    paper=paper.replace(i,' ')
                #print line
                #split sentences
                line=[]
                sentences=paper.split('.')
                for one_sen in sentences:
                    array=one_sen.split(' ')
                    while '' in array:
                        array.remove('')
                    line.append(array)
                doc.append(line)
        print 'read doc complete'
        #remove common words and words that only appear once
        frequency = defaultdict(int)
        stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
        for line in doc:
            for sent in line:
                for word in sent:
                    frequency[word]+=1
        doc=[[[word for word in sent if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
                for sent in line]for line in doc]  

        #remove sentences that are too short
        for i in xrange(len(doc)):
            for j in xrange(len(doc[i])-1,-1,-1):
                if len(doc[i][j])<2:
                    doc[i].pop(j)
        #print doc

        #cut the line if it is too long
        max_sent=15
        max_words_per_sent=15
        for i in xrange(len(doc)):
            if len(doc[i])>max_sent:
                doc[i]=doc[i][0:max_sent]
            for j in xrange(len(doc[i])):
                if len(doc[i][j])>max_words_per_sent:
                    doc[i][j]=doc[i][j][0:max_words_per_sent]
        print 'preprocess doc complete'

        dic=self.sort_word(doc)#dictionary from word to index
        #add word dictionary. doc will only save indics
        #dic={} #dictionary from word to index
        doc_index=[]#same size as doc. but it save indics instead of word
        index=0
        for paper in doc:
            paper_index=[]
            for sent in paper:
                sent_index=[]
                for word in sent:
                    sent_index.append(dic[word])
                paper_index.append(sent_index)
            doc_index.append(paper_index)
        #print doc_index

        #convert list into array
        #get mask for each sent
        doc_array=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.int32)
        doc_array_reverse=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.int32)
        mask=np.zeros([len(doc_index),max_sent],dtype=np.int32)
        mask_bool=np.zeros([len(doc_index),max_sent,max_words_per_sent],dtype=np.float32)
        for i in xrange(len(doc_index)):
            for j in xrange(len(doc_index[i])):
                mask[i][j]=len(doc_index[i][j])
                for k in xrange(len(doc_index[i][j])):
                    doc_array[i][j][k]=doc_index[i][j][k]
                    doc_array_reverse[i][j][mask[i][j]-1-k]=doc_index[i][j][k]
                    mask_bool[i][j][k]=1
        print 'get mask complete'
        
        #get word vectors
        model=Word2Vec.load(model_path)
        word_vec=np.zeros([len(dic)+1,model['computer'].shape[0]],dtype=np.float32) #size:word_num*vector_per_word
        for word in dic:
            vec=model[word]
            word_vec[dic[word],:]=vec

        # doc_array=doc_array[self.new_item_index]
        # doc_array_reverse=doc_array_reverse[self.new_item_index]
        # mask=mask[self.new_item_index]
        # mask_bool=mask_bool[self.new_item_index]
        print 'read text data complete.'
        return doc_array,doc_array_reverse,mask,mask_bool,word_vec
