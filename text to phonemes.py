import nltk #nltk = natural language tool kit
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
pronunciations = cmudict.dict() #this makes the function easier to call

def combine(words):  #This function recombines contractions and takes out commas
    x = []
    for w in words:
        if w != "," and "'" not in w:
            x.append(w)
        elif "'" in w:
            x[len(x)-1] += w
    
    return(x)

def process_content():
    while True:
        input_text = input("Type a sentance:")
        input_text = input_text.lower()
        input_text = input_text.replace(".", " ")
    
        try:
            words = word_tokenize(input_text)
            words = combine(words)
            print(words)
            for w in words:
                if len(pronunciations[w]) > 1:
                    print ((pronunciations[w])[0])
                else:
                    print(pronunciations[w])    
        except Exception as e:
            print (str(e))
            
process_content()
