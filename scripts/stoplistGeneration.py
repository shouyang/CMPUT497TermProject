import loadData
import spacy
from spacy.matcher import PhraseMatcher
import time
from prettytable import PrettyTable


class StoplistWord:
    '''Class used to keep track of frequencies of each adjacent word'''

    def __init__(self,word):
        self.word = word
        self.termFreq = 0
        self.docFreq = 0
        self.adjFreq = 0
        self.keywordFreq = 0

def genTokenizedText(abstract):
    """ Helper function, generates the tokenized version of the abstract
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(abstract.lower())

    return doc

def getAdjacencyWords (abstract,tokenizedText):
    """ Helper function, extracts the adjacent words of each keyword in the abstract
    """

    #adapted from https://spacy.io/usage/rule-based-matching
    nlp = spacy.load('en_core_web_sm')

    adjacentWords = {}

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms = abstract.actual_keywords
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("TerminologyList", None, *patterns)

    doc = tokenizedText
    matches = matcher(doc)
    # print(abstract.abstract)
    # print("\n")
    # print(terms)

    for match_id, start, end in matches:
        span = doc[start:end]
        # leftword = doc[start-1]
        # rightword = doc[end]

    #    print(span)

        left_i = start - 1
        right_i = end

        while left_i >= 0:
            if doc[left_i].is_alpha or doc[left_i].is_digit:
                if doc[left_i] not in adjacentWords:
                    adjacentWords[doc[left_i].text.lower()] = 1
                else:
                    adjacentWords[doc[left_i].text.lower()] += 1
                break
            else:
                left_i -= 1    

        while right_i < len(doc):
            if doc[right_i].is_alpha or doc[right_i].is_digit:
                if doc[right_i] not in adjacentWords:
                    adjacentWords[doc[right_i].text.lower()] = 1
                else:
                    adjacentWords[doc[right_i].text.lower()] += 1
                break
            else:
                right_i += 1                         

    return adjacentWords

def getFrequencies (stopwordObjList,abstractObj,tokenizedText,adjacentWords):
    '''See if the adjacent word currently exists in the stoplist list. if it does, update the values for it. if not, create a new StoplistWord object '''

    nlp = spacy.load('en_core_web_sm')

    for adjacentWord in adjacentWords:

        #get matches of each adjacency word from the entire abstract
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        terms = adjacentWords
        patterns = [nlp.make_doc(text) for text in terms]
        matcher.add("TerminologyList", None, *patterns)

        doc = tokenizedText
        matches = matcher(doc)
        matches = [match for match in matches]

        termfreq = 0 
        docfreq = 0
        adjfreq = 0
        keywordfreq = 0

        #TODO cite this in paper https://stackoverflow.com/questions/1801668/convert-a-python-list-with-strings-all-to-lowercase-or-uppercase        

        #get term freq
        for match_word,start,end in matches:
            matchInText = doc[start].text.lower()
            if adjacentWord == matchInText:
                termfreq +=1

        #get doc freq
        docfreq = 1

        #get adj freq
        adjfreq = adjacentWords.get(adjacentWord)

        #get keyword freq, reset Matcher to look in the keywordlist instead of abstract
        for keyword in abstractObj.actual_keywords:
            doc = nlp(keyword)
            matches = matcher(doc)
            matches = [match for match in matches]

            for match_word,start,end in matches:
                matchInText = doc[start].text.lower()
                if adjacentWord == matchInText:
                    keywordfreq +=1

        if len(stopwordObjList)>0: #handle first iteration of function
            for stopListObject in stopwordObjList:
                
                stopwordExists = False

                if stopListObject.word == adjacentWord: #the object for this word already exists, so just update the counts
                    stopListObject.termFreq += termfreq
                    stopListObject.docFreq += docfreq
                    stopListObject.adjFreq += adjfreq
                    stopListObject.keywordFreq += keywordfreq

                    stopwordExists = True
                    break

            if not stopwordExists:
                createStopWordObject (stopwordObjList,adjacentWord,termfreq,docfreq,adjfreq,keywordfreq)

        else: #create new stoplist object and add it to the stopwordlist
            createStopWordObject (stopwordObjList,adjacentWord,termfreq,docfreq,adjfreq,keywordfreq)            
       
    return stopwordObjList

def createStopWordObject (stopwordObjList,adjacentWord,termfreq,docfreq,adjfreq,keywordfreq):

    #initialize new stopword object and append it to the stopword object list

    stoplistword = StoplistWord(adjacentWord)
    stoplistword.termFreq = termfreq
    stoplistword.docFreq = docfreq
    stoplistword.adjFreq = adjfreq
    stoplistword.keywordFreq = keywordfreq

    stopwordObjList.append(stoplistword)

def createStoplist (stopwordObjList):

    stoplistFile = open("keywordAdjacencyStoplist.txt", 'w',encoding="utf-8")
    excludedFile = open("excludedFromStoplist.txt", 'w',encoding="utf-8")
    table1_3 = open("table1_3.txt", 'w',encoding="utf-8")

    stoplistList = []
    excludedList = []

    #table1_3.write("Word" + "\t"+"\t" + "Document Freq" + "\t"+"\t" + "Adjacency Freq" + "\t"+"\t" + "Keyword Freq" + "\n")

    table1_3table = PrettyTable(["Word", "Term Freq", "Document Freq","Adjacency Freq","Keyword Freq"])

    #if the keyword frequency is higher than the adjacency frequency, don't add it to the stoplist
    for stopWord in stopwordObjList:
        if stopWord.keywordFreq > stopWord.adjFreq:
            excludedList.append(stopWord.word)
            excludedFile.write(stopWord.word + "\n")
            table1_3table.add_row([stopWord.word, str(stopWord.termFreq), str(stopWord.docFreq),str(stopWord.adjFreq),str(stopWord.keywordFreq)])
        elif stopWord.termFreq >= 10:
#        else:
            stoplistList.append(stopWord.word)
            stoplistFile.write(stopWord.word + "\n")
            table1_3table.add_row([stopWord.word, str(stopWord.termFreq), str(stopWord.docFreq),str(stopWord.adjFreq),str(stopWord.keywordFreq)])
   
    table1_3.write(table1_3table.get_string())
    
    stoplistFile.close()
    excludedFile.close()
    table1_3.close()

    return stoplistList,excludedList

   
if __name__ == "__main__":

    start_time = time.time()

    #initialize frequency dictionary
    stopwordObjList = []
    
    abstracts = loadData.getAbstracts(True)
    print(len(abstracts))

    for a in abstracts:

        tokenizedText = genTokenizedText(a.abstract)
        adjacentWords = getAdjacencyWords(a,tokenizedText)
        stopwordObjList = getFrequencies(stopwordObjList,a,tokenizedText,adjacentWords)
        
        # print(a.abstract)

        # print(len(stopwordObjList))

        # for eachitem in stopwordObjList:
        #     print(eachitem.word)
        #     print("adjfreq",eachitem.adjFreq)
        #     print("docfreq",eachitem.docFreq)
        #     print("keywordfreq",eachitem.keywordFreq)
        #     print("termfreq",eachitem.termFreq)
        #     print("\n")

#        x = input()
#        if x:
#            break

    stoplistList,excludedList = createStoplist(stopwordObjList)
    print(stoplistList)
    print(excludedList)

    print("--- %s minutes ---" % ((time.time() - start_time)/60))