import subprocess
import json
from collections import defaultdict
f = open ('containfile.json', "r")
top50Articles = json.loads(f.read())
f.close()

# Reading from file
sentences = defaultdict(list)
sentencesFiltered = defaultdict(list)
top50ArticlesReduced = defaultdict(list)
#sentenceList = 
full = False
def splitIntoSentence(frequency):
    
    for word in frequency:
        full = False
        currentArticlesLength = min(len(frequency[word]), 50)
        #print(len(frequency[word]), 50)
        for idx in range(currentArticlesLength): ### only pick 50 articles; change it to 50; be careful, it is 50 articles; not 50 sentences
            if full == True: break
            file = frequency[word][idx]
            fileContailWord = False
            #print(word, frequency[word][idx], file)
            with open(file,"r",encoding='utf-8') as currfile:
                for line in currfile.readlines(): 
                    if full == True: break                       
                    if '"text":' in line:
                        a, b = line.split(":", 1)
                        #print(b)
                        b_list = b.split(". ")
                        for sentence in b_list:
                            if full == True: break
                            if word in sentence:
                                sentences[word].append([sentence, file])  ### changed by lixiang; this is a list of sentences
                                if len(sentences[word]) > 50: 
                                    full = True
                                    break
                                
                                fileContailWord = True
                                break
                        if fileContailWord == True: break

                        '''
                        ### debug some sentence might have word size that is larger than 512
                        if word == "he": 
                            #print(len(sentences[word ]),b.split(". "))
                            print(word, len(sentences[word]), sentences[word])
                        '''
                        #print(word)
                        #print(sentences)
    
    
    for word in sentences:
        for idx in range(len(sentences[word])):
            
            
            if " " + word + " " in sentences[word][idx][0] and len(sentences[word][idx][0].split(" ")) <= 512: ### this is to make sure that for example "nurse" is in the sentence, not nursery
                sentencesFiltered[word].append(sentences[word][idx][0])
                if sentences[word][idx][1] not in top50ArticlesReduced[word]:
                    top50ArticlesReduced[word].append(sentences[word][idx][1])
    #print(sentencesFiltered)
                
splitIntoSentence(top50Articles)
#quit()

# In[5]:

filteredSentenceFile = "./sentenceFile.json"
containfileReduced = "./containfileReduced.json"
def writeFilteredSentence(sentencesFiltered):

    with open(filteredSentenceFile, "w", encoding='utf-8') as filteredSentence:
        sentenceFildered_Object = json.dumps(sentencesFiltered, indent = 4)
        filteredSentence.write(sentenceFildered_Object)

    with open(containfileReduced, "w", encoding='utf-8') as containfileRe:
        top50ArticlesReduced_Object = json.dumps(top50ArticlesReduced, indent = 4)
        containfileRe.write(top50ArticlesReduced_Object)    
        #pass

writeFilteredSentence(sentencesFiltered)


#quit()