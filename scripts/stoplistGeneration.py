import loadData
import spacy
from spacy.matcher import PhraseMatcher

def genTokenizedText(abstract):
    """ Helper function, generates the 
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(abstract)

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
    for match_id, start, end in matches:
        span = doc[start:end]
        leftword = doc[start-1]
        rightword = doc[end]

        for word in [leftword,rightword]:
            if word.is_alpha:
                if word not in adjacentWords:
                    adjacentWords[word.text.lower()] = 1
                else:
                    adjacentWords[word.text.lower()] += 1 

    return adjacentWords

def getFrequencies (stoplistList,abstractObj,tokenizedText,adjacentWords):
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
        print(adjacentWord,termfreq)

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

        if len(stoplistList)>0: #handle first iteration of function
            for stopListObject in stoplistList:
                
                stopwordExists = False

                if stopListObject.word == adjacentWord: #the object for this word already exists, so just update the counts
                    stopListObject.termFreq += termfreq
                    stopListObject.docFreq += docfreq
                    stopListObject.adjFreq += adjfreq
                    stopListObject.keywordFreq += keywordfreq
                    print("this word exists in the list",stopListObject.word)

                    stopwordExists = True
                    break

            if not stopwordExists:
                # stoplistword = StoplistWord(adjacentWord)
                # stoplistword.termFreq = termfreq
                # stoplistword.docFreq = docfreq
                # stoplistword.adjFreq = adjfreq
                # stoplistword.keywordFreq = keywordfreq

                # stoplistList.append(stoplistword)

                createStopWordObject (stoplistList,adjacentWord,termfreq,docfreq,adjfreq,keywordfreq)

        else: #create new stoplist object and add it to the stopwordlist
            # stoplistword = StoplistWord(adjacentWord)
            # stoplistword.termFreq = termfreq
            # stoplistword.docFreq = docfreq
            # stoplistword.adjFreq = adjfreq
            # stoplistword.keywordFreq = keywordfreq

            # stoplistList.append(stoplistword)

            createStopWordObject (stoplistList,adjacentWord,termfreq,docfreq,adjfreq,keywordfreq)            
       
    return stoplistList

def createStopWordObject (stoplistList,adjacentWord,termfreq,docfreq,adjfreq,keywordfreq):

    stoplistword = StoplistWord(adjacentWord)
    stoplistword.termFreq = termfreq
    stoplistword.docFreq = docfreq
    stoplistword.adjFreq = adjfreq
    stoplistword.keywordFreq = keywordfreq

    stoplistList.append(stoplistword)

def createStoplist ():
    pass


#initialize frequency dictionary
stoplistList = []

class StoplistWord:
    '''Class used to keep track of frequencies of each adjacent word'''

    def __init__(self,word):
        self.word = word
        self.termFreq = 0
        self.docFreq = 0
        self.adjFreq = 0
        self.keywordFreq = 0

    

for a in loadData.getAbstracts():
    tokenizedText = genTokenizedText(a.abstract)
    adjacentWords = getAdjacencyWords(a,tokenizedText)
    # print(a.abstract)
    # print(a.actual_keywords)
    # print(adjacentWords)
    stoplistList = getFrequencies(stoplistList,a,tokenizedText,adjacentWords)
    
    print(a.abstract)

    print(len(stoplistList))

    for eachitem in stoplistList:
        print(eachitem.word)
        print("adjfreq",eachitem.adjFreq)
        print("docfreq",eachitem.docFreq)
        print("keywordfreq",eachitem.keywordFreq)
        print("termfreq",eachitem.termFreq)
        print("\n")

    x = input()
    if x:
        break
