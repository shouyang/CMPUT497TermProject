import loadData
import spacy
from spacy.matcher import PhraseMatcher


def getAdjacencyWords (abstract):
    """ Helper function, extracts the adjacent words of each keyword in the abstract
    """

    #adapted from https://spacy.io/usage/rule-based-matching
    nlp = spacy.load('en_core_web_sm')

    adjacentWords = {}

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms = abstract.actual_keywords
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("TerminologyList", None, *patterns)

    doc = nlp(abstract.abstract)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        leftword = doc[start-1]
        rightword = doc[end]

        for word in [leftword,rightword]:
            if word.is_alpha:
                if word not in adjacentWords:
                    adjacentWords[word.text] = 1
                else:
                    adjacentWords[word.text] += 1 

    return adjacentWords

def getFrequencies ():
    pass

def createStoplist ():
    pass


for a in loadData.getAbstracts():
    print(getAdjacencyWords(a))

    x = input()
    if x:
        break
