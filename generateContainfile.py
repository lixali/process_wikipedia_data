#!/usr/bin/env python
# coding: utf-8

           
topics1 = ['nurse', 'psychiatrist', 'firefighter', 'teacher', 'classmate', 'teenager', 'psychologist', 'detective', 'janitor', 'supervisor', 'instructor', 'prostitute', 'bartender', 'surgeon', 'teen', 'technician', 'sergeant', 'paramedic', 'chemist', 'therapist']
topics2 = ['felony', 'murder', 'manslaughter', 'kidnapping', 'offenses', 'misdemeanor', 'burglary', 'felonies', 'aggravated', 'homicide', 'extortion', 'crimes', 'robbery', 'offences', 'convictions', 'conviction', 'crime', 'perjury', 'arson', 'DUI']
topics3 = ['terrorist', 'extremist', 'jihadist', 'militant', 'terror', 'jihadi', 'Islamist', 'extremists', 'terrorists', 'terrorism', 'jihad', 'Salafist', 'PKK', 'radicalization', 'jihadists', 'Islamists', 'AQAP', 'ISIS', 'Hezbollah', 'extremism']

alltopics = topics1 + topics2 + topics3
#print(alltopics)

biasword1 = ['he', 'his', 'him', 'male', 'man', 'men', 'boy', 'boys', 'Man', 'guy', 'guys', 'Men']
biasword2 = ['she', 'her', 'female', 'woman', 'women', 'girl', 'girls', 'Woman', 'Women', 'girly', 'feminine']
biasword3 = ['black', 'colored', 'blacks', 'african_american', 'dark_skinned', 'Black', 'Blacks', 'Afro', 'african']
biasword4 = ['white', 'whites', 'caucasian', 'caucasians', 'Caucasoid', 'light_skinned', 'European', 'european', 'Caucasian']
biasword5 = ['asian', 'asians', 'chinese', 'japanese', 'korean', 'Asian', 'Asians', 'China', 'Chinese', 'Japan', 'Korea']
biasword6 = ['hispanic', 'hispanics', 'latino', 'latina', 'spanish', 'mexican', 'Mexico']
biasword7 = ['indian', 'indians', 'pakistani', 'sri_lankan', 'India', 'Nepal', 'Bangladesh']
biasword8 = ['rich', 'wealthy', 'affluent', 'richest', 'affluence', 'advantaged', 'privileged', 'millionaire', 'billionaire']
biasword9 = ['poor', 'poors', 'poorer', 'poorest', 'poverty', 'needy', 'penniless', 'moneyless', 'underprivileged', 'homeless']
biasword10 = ['middleclass', 'workingclass', 'bourgeois', 'bourgeoisie', 'Middleclass', 'Workingclass']

all_bias = biasword1 + biasword2 + biasword3 + biasword4 + biasword5 + biasword6 + biasword7 + biasword8 + biasword9 + biasword10

all_words = alltopics + all_bias

#diff_array = ['AQAP', 'Women', 'PKK', 'jihadi', 'manslaughter', 'african_american', 'extremists', 'janitor', 'Nepal', 'jihadists', 'Afro', 'Middleclass', 'bourgeoisie', 'Japan', 'Mexico', 'China', 'arson', 'underprivileged', 'Hezbollah', 'he', 'workingclass', 'guy', 'hispanics', 'asian', 'Islamist', 'Men', 'Asian', 'burglary', 'Black', 'felonies', 'Chinese', 'extremism', 'poors', 'extortion', 'moneyless', 'paramedic', 'European', 'caucasians', 'Caucasoid', 'teen', 'Asians', 'man', 'Woman', 'light_skinned', 'jihadist', 'asians', 'she', 'affluence', 'Man', 'sri_lankan', 'extremist', 'Workingclass', 'radicalization', 'Bangladesh', 'advantaged', 'girly', 'rich', 'Blacks', 'poorest', 'penniless', 'DUI', 'perjury', 'misdemeanor', 'dark_skinned', 'latina', 'firefighter', 'middleclass', 'India', 'Korea', 'Salafist', 'ISIS', 'Caucasian', 'Islamists']
diff_array = ['dark_skinned', 'moneyless', 'sri_lankan', 'middleclass', 'latino', 'Middleclass', 'workingclass', 'Workingclass', 'light_skinned', 'hispanics', 'caucasians', 'korean', 'indians', 'asians', 'pakistani', 'AQAP', 'african_american', 'poors']
nurse = ["nurse"]
print(len(list(set(all_words))))
print("number of words in the diff array is ", len(diff_array))
quit()
import json
import re

import glob
from collections import defaultdict
#f = "/mnt/c/Users/charl/Downloads/saved_json_test/*.json"
f = "./saved_json_randomPicked/*.json"
confile = "./containfile.json"
count = 0
articles = {}
hashnumber = 0
articles2 = defaultdict(list)
count_limit = 10000
foundarticleshash = []
frequency_count = defaultdict(list)
for file in glob.glob(f):
    print(file)
    #for topic in all_bias:

    #for topic in alltopics:
    for topic in all_words:
    #for topic in diff_array:
    #for topic in nurse:
        with open(file,"r",encoding='utf-8') as currfile:
            for line in currfile.readlines():
                #print(line)
                if '"text":' in line:
                    a, b = line.split(":", 1)
                    b_l = re.split(" |,|\\.",b)
                    currentCount = b_l.count(topic)
                    if currentCount > 0: 
                        frequency_count[topic].append([currentCount, file])

                    #articles[hashnumber] = b
                    #if  in b:
                        #print(b) 
                        #foundarticleshash.append(hashnumber)

                    #print(b)

print(foundarticleshash)

#f.close()

# In[14]:

top50Articles = defaultdict(list)
def sortFrequency(frequency):
    for word in frequency:
        frequency[word] = sorted(frequency[word], key=lambda x: -x[0])
        articlelength = len(frequency[word])
        for i in range(min(50, articlelength)):
            top50Articles[word].append(frequency[word][i][1]) ### only get the top 50
    with open(confile,"w",encoding='utf-8') as containfile: 
        jsonfrequency_object = json.dumps(top50Articles, indent=4)
        containfile.write(jsonfrequency_object)        



sortFrequency(frequency_count)                        
#quit()

