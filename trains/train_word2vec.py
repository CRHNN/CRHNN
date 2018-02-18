import numpy as np
import string
import csv
from collections import defaultdict
import gensim
from gensim.models import Word2Vec
data_path='/home/cc/data/ctrsr_datasets/citeulike-t/rawtext.dat'
citeulike_a_path='/home/cc/data/ctrsr_datasets/citeulike-a/raw-data.csv'
movielens_path='/home/cc/data/ctrsr_datasets/movielens_plot_data/movielens_1m/movies.plot'
#model_path='/home/chenchen/recommendation/data/citeulike-t_word2vec_model_32'
model_path='/home/chenchen/recommendation/data/movielens_word2vec_model_32'
#model_path='/home/chenchen/recommendation/data/word2vec_test'
def read_movielens_1m(text_path):
    doc=[]
    punctuation='!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    nextline='\n'
    with open(text_path, "r") as f:
        paper = f.readline()
        while paper != None and paper != "":
            paper=paper.split('::')
            paper=paper[2]
            paper=paper.lower()
            for i in string.punctuation:
                paper=paper.replace(i,' ')
            for i in nextline:
                paper=paper.replace(i,' ')
            #print line
            #split sentences
            array=paper.split(' ')
            while '' in array:
                array.remove('')
            #print array
            doc.append(array)
            paper=f.readline()
            
    #remove common words and words that only appear once
    frequency = defaultdict(int)
    stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
    for line in doc:
        for word in line:
            frequency[word]+=1
    doc=[[word for word in line if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
            for line in doc]
    print 'read text data complete.'
    return doc
def read_citeulike_t(text_path):
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
                for i in string.punctuation:
                    paper=paper.replace(i,' ')
                for i in nextline:
                    paper=paper.replace(i,' ')
                #print line
                #split sentences
                array=paper.split(' ')
                while '' in array:
                    array.remove('')
                #print array
                doc.append(array)
                paper=f.readline()
            else:
                text=True
                paper=f.readline()
            
    #remove common words and words that only appear once
    frequency = defaultdict(int)
    stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
    for line in doc:
        for word in line:
            frequency[word]+=1
    doc=[[word for word in line if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
            for line in doc]
    print 'read text data complete.'
    #print doc
    return doc
def read_citeulike_a(path):
    doc=[]
    with open(path,'r') as f:
        reader=csv.reader(f)
        first=1
        for row in reader:
            if first>0:
                first=0
                continue
            line=row[1].lower()+' '+row[4].lower()
            #remove punctuation marks
            for i in '{}':
                line=line.replace(i,'')
            for i in string.punctuation:
                line=line.replace(i,' ')
            #print line
            array=line.split(' ')
            while '' in array:
                array.remove('')
            #print array
            doc.append(array)
    #remove common words and words that only appear once
    frequency = defaultdict(int)
    stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
    for line in doc:
        for word in line:
            frequency[word]+=1
    doc=[[word for word in line if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
            for line in doc]  
    print 'read text data complete.'
    #print doc
    return doc
def train():
    pass
    #doc=read_citeulike_a(citeulike_a_path)
    #doc=read_citeulike_t(data_path)
    doc=read_movielens_1m(movielens_path)
    model=Word2Vec(doc,size=32,workers=4)
    model.save(model_path)
def main():
    train()


if __name__ == '__main__':
    main()