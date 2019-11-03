import os
from bs4 import BeautifulSoup
import json


class Abstract:
    """ Class used to denote a research paper abstract to be used as sample for evaluating keyword extraction methods.
    """
    def __init__(self, filepath, keyword_reference):
        # Absolute Filepath used to create this object.
        self.filepath = os.path.abspath(filepath)
        # File name of this item, should be an integer
        self.num      = os.path.splitext(os.path.basename(filepath))[0]
        # BeautifulSoup representation of the file 
        self.doc      = BeautifulSoup(open(self.filepath, "r", encoding="utf-8").read(), "xml")
        # String-paragraph denoting the actual contents of the file.
        self.sentence_tokens, self.abstract = self.gen_abstract()
        # Actual keywords assigned to abstract from the references file (uncontrolled references).
        self.actual_keywords = self.gen_actual_keywords(keyword_reference)

    def gen_abstract(self):
        """ Generates the string representation of the abstract from the XML file.
        """

        # Get (word, POS-tag) pairs from the docment. Generate a list of lists of sentences -> 2-tuples.
        sentence_tokens = []
        for sent in self.doc.find_all("sentence"):
            sent_words = []
            for token in sent.find_all("token"):
                sent_words.append((token.word.text, token.POS.text))
            sentence_tokens.append(sent_words)

        # Reconstruct original paragraph via tokens, add or remove spaces, and puncuation as required.
        PUNCUATION_POS_TAGS = [",", ".", ":"]
        REPLACEMENTS  = {
            "-LRB-": " (",
            "-RRB-": ")"
        }
        PREV_CHAR_TO_NOT_ADD_SPACES_TO = ["("]

        output = ""
        for sentence_tuple_list in sentence_tokens:
            sentence = ""
            for word, POS_tag in sentence_tuple_list:
    
                # Do not add spaces for puncuation.
                if POS_tag in PUNCUATION_POS_TAGS:
                    sentence += word

                # Add certain things from original document.
                elif word in REPLACEMENTS:
                    sentence += REPLACEMENTS[word]
                    continue
    
                else:
                    if sentence:
                        if sentence[-1] in PREV_CHAR_TO_NOT_ADD_SPACES_TO:
                            sentence += word
                        else:
                            sentence += " " + word
                    else:
                        sentence += word

            # Some sentences do not actually end on a period. Add if necessary.
            if sentence[-1] != ".":
                sentence += ". "
            if sentence[-1] == ".":
                sentence += " "

            output += sentence

        # Output both the parsed series of tokens (for debug) and the actual abstract paragraph.
        return (sentence_tokens, output)

    def gen_actual_keywords(self,keyword_reference):
        """ Helper function, extracts the "true" keywords from the keyword reference JSON file object.
        """
        output = []
        for nested_list in keyword_reference[self.num]:
            assert(len(nested_list) == 1)
            output.append(nested_list[0])

        return output



def getAbstracts():
    """ Loader function, used to get object representations of all of the test data-folder abstracts.
    """
    JSON_FILE_PATH = r".\data\references\test.uncontr.json"  
    XML_FILE_PATH  = r".\data\test"

    keyword_reference = json.load(open(JSON_FILE_PATH, "r", encoding="utf-8"))

    abstracts = []
    for  root, dirs, files in os.walk(XML_FILE_PATH):
        for f in files:
            fullpath = os.path.join(root, f)
            if fullpath.endswith(".xml"):
                abstracts.append(Abstract(fullpath, keyword_reference))

    return abstracts

if __name__ == "__main__":
    for a in getAbstracts():
        print(a.filepath)
        print(a.num)
        print(a.abstract)
        print(a.actual_keywords)
        
        x = input()
        if x:
            break