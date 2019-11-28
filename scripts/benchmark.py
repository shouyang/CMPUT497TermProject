import nltk
import statistics

from loadData    import getAbstracts
from rake_nltk   import Rake
import pytextrank


FOX_SL= ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'because', 'become', 'becomes', 'became', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'herself', 'here', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'no', 'non', 'not', 'nobody', 'noone', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'sees', 'seem', 'seemed', 'seeming', 'seems', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'uses', 'used', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours']

KA_stoplist=['new', 'the', 'by', 'that', 'one', 'of', 'is', 'called', 'instead', 'with', 'for', 'in', 'provide', 'are', 'such', 'as', 'and', 'it', 'on', 'we', 'a', 'provides', 'an', 'this', 'or', 'however', 'where', 'these', 'have', 'because', 'which', 'to', 'along', 'using', 'both', 'scheme', 'multiple', 'its', 'has', 'method', 'more', 
'many', 'based', 'like', 'effective', 'thus', 'from', 'about', 'during', 'our', 'process', 'also', 'without', 'techniques', 'associated', 'three', 'simple', 'technique', 'over', 'through', 'after', 'any', 'among', 'existing', 'application', 'derived', 'may', 'were', 'was', 'than', 'under', 'results', 'between', 'while', 'but', 'uses', 'used', 'into', 'can', 'resulting', 'accurate', 'simulation', 'different', 'approach', 'design', 'applied', 'at', 'including', 'performance', 'proposed', 'must', 'system', 'model', 'systems', 'conventional', 
'then', 'order', 'methods', 'each', 'standard', 'language', 'high', 'improved', 'when', 'other', 'their', 'ii', 'make', 'if', 'two', 'how', 'study', 'single', 'several', 'only', 'various', 'given', 'via', 'general', 'use', 'c', 'consists', 'applications', 'known', 'some', 'include', 'most', 'all', 'research', 'participation', 'computer', 'code', 'architecture', 'services', 'first', 'x', 'problems', 'p', 'cbr', 'case', 'sub', 'respectively', 'cm', 'imrt', 'dimensional', 'sup', 'algorithm', 'there', 'developed', 'large', 'will', 'traditional', 'efficient', 'current', 'time', 'hpf', 'data', 'programs', 'models', 'what', 'access', 'problem', 'judgment', 'building', 'manufacturing', 'n', 'hybrid', 'technology', 'content', 'ga', 'trained', 'mlp']

top_100_words_stoplist = ["the", "and", "of", "a", "in", "is", "for", "to", "we", "this", "are", "with", "as", "on", "it", "an", "that", "which", "by", "using", "can", "paper", "from", "be", "based", "has", "was", "have", "or", "at", "such", "also", "but", "results", "proposed", "show", "new", "these", "used", "however", "our", "were", "when", "one", "not", "two", "study", "present", "its", "sub", "both", "then", "been", "they", "all", "presented", "if", "each", "approach", "where", "may", "some", "more", "use", "between", "into", "1", "under", "while", "over", "many", "through", "addition", "well", "first", "will", "there", "propose", "than", "their", "2", "most", "sup", "developed", "particular", "provides", "including", "other", "how",
"without", "during", "article", "application", "only", "called", "what", "since", "order", "experimental", "any"]

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

    return 1.0 / (alpha / precision + (1 - alpha) / recall)


def get_RAKE_keywords(sample, stoplist, T_DENOM = 3):

    rake = Rake(stopwords = stoplist)
    rake.extract_keywords_from_text(sample.abstract)

    t = len(rake.get_word_degrees()) // T_DENOM

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

    
abstracts = getAbstracts()
counts = {}

for sample in abstracts:
    # Get Actual Keywords
    actual_keywords           = sample.actual_keywords
    
    # Get Estimates
    rake_estimate_keywords      = get_RAKE_keywords(sample, FOX_SL)
    textrank2_estimate_keywords = get_TextRank_keywords(sample, window = 2)
    textrank3_estimate_keywords = get_TextRank_keywords(sample, window = 3)
    
    # Tabluate Estimates
    tabluate("RAKE-FOX", rake_estimate_keywords, actual_keywords, counts)
    tabluate("TEXT-2", textrank2_estimate_keywords, actual_keywords, counts)
    tabluate("TEXT-3", textrank3_estimate_keywords, actual_keywords, counts)

    # Generate Output
    # print("Abstract:", sample.abstract)
    # print("RAKE:", rake_estimate_keywords)
    # print("TextRank WINDOW = 2: ", textrank2_estimate_keywords)
    # print("TextRank WINDOW = 3: ", textrank3_estimate_keywords)
    # print("Actual Keywords: ", actual_keywords)
    
    # x = input()
    # if x:
        # break

print(counts[("RAKE-FOX", "Assigned")])
print(counts[("RAKE-FOX", "Correct")])
print(statistics.mean(counts[("RAKE-FOX","Precision")]))
print(statistics.mean(counts[("RAKE-FOX","Recall")]))
print("--")
print(counts[("TEXT-2", "Assigned")])
print(counts[("TEXT-2", "Correct")])
print(statistics.mean(counts[("TEXT-2","Precision")]))
print(statistics.mean(counts[("TEXT-2","Recall")]))
print("--")
print(counts[("TEXT-3", "Assigned")])
print(counts[("TEXT-3", "Correct")])
print(statistics.mean(counts[("TEXT-3","Precision")]))
print(statistics.mean(counts[("TEXT-3","Recall")]))