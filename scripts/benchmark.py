import nltk
import statistics

from loadData    import getAbstracts
from rake_nltk   import Rake
import pytextrank


FOX_SL= ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'because', 'become', 'becomes', 'became', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'herself', 'here', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'no', 'non', 'not', 'nobody', 'noone', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'sees', 'seem', 'seemed', 'seeming', 'seems', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'uses', 'used', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours']
KA_doc_freq_stoplist = ['new', 'the', 'by', 'that', 'one', 'of', 'is', 'called', 'with', 'for', 'in', 'are', 'such', 'as', 'and', 'it', 'on', 'we', 'a', 'provides', 'an', 'this', 'or', 'however', 'where', 'these', 'have', 'because', 'which', 'to', 'along', 'using', 'both', 'its', 'has', 'method', 'more', 'many', 'based', 'thus', 'from', 'during', 'our', 'process', 'also', 'without', 'techniques', 'associated', 'three', 'simple', 'technique', 'over', 'through', 'after', 'any', 'among', 'may', 'were', 'was', 'than', 'under', 'between', 'while', 'but', 'uses', 'used', 'into', 'can', 'different', 'approach', 'at', 'including', 'proposed', 'must', 'system', 'model', 'systems', 'conventional', 'methods', 'each', 'standard', 'when', 'other', 'their', 'if', 'two', 'how', 'study', 'several', 'various', 'given', 'via', 'use', 'some', 'all', 'first', 'will', 'efficient', 'data']

def get_recall(estimate, actual):
    estimate = set(estimate)
    actual   = set(actual)

    assert(len(estimate) != 0)
    assert(len(actual) != 0)

    return len(estimate.intersection(actual)) / len(actual) * 1.0

def get_precision(estimate, actual):
  
    estimate = set(estimate)
    actual   = set(actual)

    assert(len(estimate) != 0)
    assert(len(actual) != 0)

    return len(estimate.intersection(actual)) / len(estimate) * 1.0

def get_F_score(estimate, actual, alpha = 0.5):
    
    precision = get_precision(estimate, actual)
    recall    = get_recall(estimate, actual)

    if precision == 0 or recall == 0:
        return 0
    else:
        return 1.0 / (alpha / precision + (1 - alpha) / recall)


def get_RAKE_keywords(sample, stoplist, T_RATIO = 0.33):

    rake = Rake(stopwords = stoplist)
    rake.extract_keywords_from_text(sample.abstract)

    t = int(len(rake.get_word_degrees()) * T_RATIO)

    return rake.get_ranked_phrases()[:t]

def get_TextRank_keywords(sample, window = 2):
    return pytextrank.top_keywords_sentences(sample.abstract, phrase_limit= None, window = window)[1]


def tabluate(method_name, estimate_keywords, actual_keywords, output):
    # Update Assigned Keywords
    key = (method_name, "Assigned")
    output[key] = output.get(key, 0) + len(estimate_keywords)

    # Update Total Correct Keywords
    estimate = set(estimate_keywords)
    actual   = set(actual_keywords)

    correct_keywords = len(estimate.intersection(actual))

    key = (method_name, "Correct")
    output[key] = output.get(key, 0) + correct_keywords

    # Update Precision
    key = (method_name, "Precision")

    temp = output.get(key, [])
    temp.append(get_precision(estimate_keywords, actual_keywords))
  
    output[key] = temp

    # Update Recall
    key = (method_name, "Recall")

    temp = output.get(key, [])
    temp.append(get_recall(estimate_keywords, actual_keywords))

    output[key] = temp

    # Update F-Score
    key = (method_name, "F-Score")

    temp = output.get(key, [])
    temp.append(get_F_score(estimate_keywords, actual_keywords))

    output[key] = temp


    
abstracts = getAbstracts()
counts = {}

fox_ka_similarity_scores              = []
textrank2_textrank3_similarity_scores = []
fox_textrank2_similarity_scores       = []

for i, sample in enumerate(abstracts):
    #print(i)
    # Get Actual Keywords
    actual_keywords           = sample.actual_keywords
    
    # Get Estimates
    rake_fox_estimate_keywords      = get_RAKE_keywords(sample, FOX_SL, 0.33)
    rake_ka_estimate_keywords       = get_RAKE_keywords(sample, KA_doc_freq_stoplist, 0.33)

    textrank2_estimate_keywords = get_TextRank_keywords(sample, window = 2)
    textrank3_estimate_keywords = get_TextRank_keywords(sample, window = 3)
    
    # Tabluate Estimates
    tabluate("RAKE-FOX", rake_fox_estimate_keywords, actual_keywords, counts)
    tabluate("RAKE-KA", rake_ka_estimate_keywords, actual_keywords, counts)
    tabluate("TEXT-2", textrank2_estimate_keywords, actual_keywords, counts)
    tabluate("TEXT-3", textrank3_estimate_keywords, actual_keywords, counts)

    # Compute and append similarity scores
    
    fox_ka_intersection = set(rake_fox_estimate_keywords).intersection(set(rake_ka_estimate_keywords))
    fox_ka_union        = set(rake_fox_estimate_keywords).union(set(rake_ka_estimate_keywords))
    fox_ka_similarity_scores.append( len(fox_ka_intersection) / len(fox_ka_union))

    textrank2_textrank3_intersection = set(textrank2_estimate_keywords).intersection(set(textrank3_estimate_keywords))
    textrank2_textrank3_union        = set(textrank2_estimate_keywords).union(set(textrank3_estimate_keywords))
    textrank2_textrank3_similarity_scores.append( len(textrank2_textrank3_intersection)/len(textrank2_textrank3_union))

    fox_textrank2_intersection = set(rake_fox_estimate_keywords).intersection(set(textrank2_estimate_keywords))
    fox_textrank2_union        = set(rake_fox_estimate_keywords).union(set(textrank2_estimate_keywords))
    fox_textrank2_similarity_scores.append(len(fox_textrank2_intersection) / len(fox_textrank2_union))

    # Print output
    #print("RAKE-FOX KEY_WORDS:", sorted(rake_fox_estimate_keywords))
    #print("TEXTRANK-3 KEY_WORDS:", sorted(textrank3_estimate_keywords))

print("RAKE-FOX")
print(counts[("RAKE-FOX", "Assigned")])
print(counts[("RAKE-FOX", "Correct")])
print("Precision: ",statistics.mean(counts[("RAKE-FOX","Precision")]))
print("Recall: ",statistics.mean(counts[("RAKE-FOX","Recall")]))
print("F-Score: ",statistics.mean(counts[("RAKE-FOX","F-Score")]))
print("Mean assigned: ",counts[("RAKE-FOX", "Assigned")]/len(abstracts))
print("Mean correct: ",counts[("RAKE-FOX", "Correct")]/len(abstracts))
print("--")
print("RAKE-KA")
print(counts[("RAKE-KA", "Assigned")])
print(counts[("RAKE-KA", "Correct")])
print("Precision: ",statistics.mean(counts[("RAKE-KA","Precision")]))
print("Recall: ",statistics.mean(counts[("RAKE-KA","Recall")]))
print("F-Score: ",statistics.mean(counts[("RAKE-KA","F-Score")]))
print("Mean assigned: ",counts[("RAKE-KA", "Assigned")]/len(abstracts))
print("Mean correct: ",counts[("RAKE-KA", "Correct")]/len(abstracts))
print("--")
print("TEXTRANK-WINDOW2")
print(counts[("TEXT-2", "Assigned")])
print(counts[("TEXT-2", "Correct")])
print(statistics.mean(counts[("TEXT-2","Precision")]))
print(statistics.mean(counts[("TEXT-2","Recall")]))
print("F-Score: ",statistics.mean(counts[("TEXT-2","F-Score")]))
print("Mean assigned: ",counts[("TEXT-2", "Assigned")]/len(abstracts))
print("Mean correct: ",counts[("TEXT-2", "Correct")]/len(abstracts))

print("--")
print("TEXTRANK-WINDOW3")
print(counts[("TEXT-3", "Assigned")])
print(counts[("TEXT-3", "Correct")])
print("Precision: ",statistics.mean(counts[("TEXT-3","Precision")]))
print("Recall: ",statistics.mean(counts[("TEXT-3","Recall")]))
print("F-Score: ",statistics.mean(counts[("TEXT-3","F-Score")]))
print("Mean assigned: ",counts[("TEXT-3", "Assigned")]/len(abstracts))
print("Mean correct: ",counts[("TEXT-3", "Correct")]/len(abstracts))

print("--")
print("SIMILARITY SCORES:")
print("FOX-KA", statistics.mean(fox_ka_similarity_scores))
print("TEXT2-TEXT3", statistics.mean(textrank2_textrank3_similarity_scores))
print("FOX-TEXT2", statistics.mean(fox_textrank2_similarity_scores))