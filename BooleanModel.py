from collections import defaultdict
import glob
import numpy as np
import re, string, unicodedata
import os
import nltk
from nltk import tokenize, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#import contractions
#import inflect
from bs4 import BeautifulSoup


from mimetypes import init


class Boolean_Model(object):
    def __init__(self, Corpus):
        self.Corpus = Corpus # path to the documents

        self.documents = dict()

        self.Terms = []
        #self.Stemmer = PorterStemmer()

        self.inverted_matix = defaultdict(list)


        self._preprocesing_corpus()


    def _strip_htlm(self,text):
        soup = BeautifulSoup(text,"html.parser")
        return soup.get_text()

    def _remove_between_square_brackets(self,text):
        return re.sub('\[[^]]*\]','',text)

    def _denoise_text(self,text):
        text = self._strip_htlm(text)
        text = self._remove_between_square_brackets(text)
        return text          

    def _denoise(self,text):
        return re.sub(r'<[^>]*?>','', text)     

    def _preprocesing_corpus(self):
        index = 1
        for filename in glob.glob(self.Corpus) :
            with open(filename,"r") as file :
                text = file.read()
            text = self._denoise(text)    
            text = self._remove_unnecesary_characters(text)
            text = self._remove_digits(text)

            #tokenization
            words = tokenize.word_tokenize(text)  

            #Remove stopwords
            words = [word.lower() for word in words if word not in stopwords.words('english')]

           # words = [self.Stemmer.stem(word) for word in words]

            terms = list(set(words)) # eliminar palabras repetidas
            for term in terms:
                self.inverted_matix[term].append(index)
                self.Terms.append(term)
            self.documents[index] = os.path.basename(filename)
            index +=1

    def _tokenizer(self,expression):
            tokens = []
            for item in expression.split():
                tokens.append(item)   
            return tokens             

    # E -> BX
    # X -> OR BX | epsilon
    # B -> AND CY 
    # Y -> AND CY |epsilon
    # C -> D | not D 
    # D -> term | (A)         

    def proces_query(self, query):
        query_tokens = self._tokenizer(query) #mejorar el tokenizer
        vector = self._evaluate_query(query_tokens)
        relevant_docs = dict()
        for i in  range(len(self.documents)):
            if vector[i] == True:
                relevant_docs[i+1] = self.documents[i+1]
        print(vector)        

        return relevant_docs

    def _evaluate_query(self,query_tokens):
        i, vector= self._parse_expression(query_tokens,0)
        if i!=len(query_tokens):
            print("La expresion no es correcta")
            return np.zeros(len(self.documents), dtype=bool)
        return vector   

    def _parse_expression(self,tokens,i):
        i,term = self._parse_B(tokens,i) 
        return self._parse_X(tokens,i,term)

    def _parse_X(self,tokens,i,value):
        if i < len(tokens):
            if tokens[i] == 'OR':
                i,term2 = self._parse_B(tokens,i+1)
                value = value | term2
                return self._parse_X(tokens,i,value)
        return i,value        

    def _parse_B(self,tokens,i):
        i,factor = self._parse_C(tokens,i)
        return self._parse_Y(tokens,i,factor)

    def _parse_Y(self,tokens,i,value):
        if i<len(tokens):
            if tokens[i] == 'AND':
                i,factor2 = self._parse_C(tokens,i+1)
                value = value & factor2
                return self._parse_Y(tokens,i,value)
        return i,value        

    def _parse_C(self,tokens,i) :
        if i < len(tokens):
            if tokens [i] == 'not':
                i, vector = self._parse_C(tokens,i+1)            
                return i, ~vector
            elif self._is_term(tokens[i]):
                vector = self._vector(tokens[i])
                return i + 1, vector
            elif tokens[i] == '(':
                i, vector = self._parse_expression(tokens,i+1) 
                if tokens[i] != ')':
                  print ("Expresion mal formada")  
                return i + 1,vector   
            else:
                print("Expresion mal formada")  
                return -1,np.zeros(len(self.documents),dtype = bool)
        else:
            print("Expresion mal formada")
            return -1,np.zeros(len(self.documents),dtype = bool)            


        

    def _is_term(self,token):
        if token == 'AND' or token == 'OR' or token == '(' or token ==')' or token == 'not':
            return False
        return True         

    def _vector(self,term):
        docs_count = len(self.documents)
        if term in self.Terms:
            vector = np.zeros(docs_count,dtype=bool)

            postings = self.inverted_matix[term]
            for doc_id in postings:
                vector[doc_id-1] = True
            return vector
        else:
            print("The term"+ term + " Was not found in the corpus")                
            return np.zeros(docs_count,dtype=bool)


    def Rank(query_vector, documents): raise NotImplementedError

    def _remove_unnecesary_characters(self,text):
        regex = re.compile(r"[^a-zA-Z0-9\s]")

        # Replace and return
        return re.sub(regex, "", text)

    def _remove_digits(self, text):
        """Removes digits from a blob of text"""

         # Regex pattern for a word
        regex = re.compile(r"\d")

        # Replace and return
        return re.sub(regex, "", text)    
     




    

            
