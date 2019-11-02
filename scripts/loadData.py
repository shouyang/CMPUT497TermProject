import os
from bs4 import BeautifulSoup
import json


class Abstract:
    def __init__(self, filepath):
        # Absolute Filepath used to create this object.
        self.filepath = os.path.abspath(filepath)
        # File name of this item, should be an integer
        self.num      = os.path.splitext(os.path.basename(filepath))[0]
        # BeautifulSoup representation of the file 
        self.doc      = BeautifulSoup(open(self.filepath, "r", encoding="utf-8").read(), "xml")
        # String-paragraph denoting the actual contents of the file.
        self.sentence_tokens, self.abstract = self.gen_abstract()
        # Actual keywords assigned to abstract from the references file (uncontrolled references).
        self.keywords = None

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
            "-LRB-": "(",
            "-RRB-": ")"
        }
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
                        sentence += " " + word
                    else:
                        sentence += word

            # Some sentences do not actually end on a period. Add if necessary.
            if sentence[-1] != ".":
                sentence += "."

            output += sentence

        # Output both the parsed series of tokens (for debug) and the actual abstract paragraph.
        return (sentence_tokens, output)



JSON_FILE_PATH = r".\data\references.test.uncontr.json"  
XML_FILE_PATH  = r".\data\test"

abstracts = []
for  root, dirs, files in os.walk(XML_FILE_PATH):
    for f in files:
        fullpath = os.path.join(root, f)
        if fullpath.endswith(".xml"):
            abstracts.append(Abstract(fullpath))

for a in abstracts:
    print(a.filepath)
    print(a.num)
    print(a.abstract)

    x = input()
    if x:
        break