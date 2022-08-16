
import json

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


f = open ('containfileReduced.json', "r")
top50Articles = json.loads(f.read())
f.close()

f2 = open ('keylist.txt', "w")

f3 = open ('embedding4.json', "r")
embedding4 = json.loads(f3.read())
f3.close()

f4 = open ('embedding4key.json', "w")

f5 = open ('sentenceFile.json', "r")
sentence = json.loads(f5.read())
f5.close()

f4 = open ('toBeDone.json', "w")

articleWord = []
for key in top50Articles:
    articleWord.append(key)

f2.write(str(articleWord))

embedkey = []
for key in embedding4:
    embedkey.append(key)

sentenceList = []
for key in sentence:
    sentenceList.append(key)

#print(len(embedkey), embedkey)
#print(len(articleWord), articleWord)
#print(len(sentenceList), sentenceList)
diff = list(set(all_words) - set(embedding4))
print("number of words that are missing is", len(diff))
print(diff)
f4.write(str(diff))
f2.close()
f4.close()
