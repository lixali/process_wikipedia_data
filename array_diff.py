
#curr = ["his", "male", "men", "boy", "boys", "her", "poverty", "chemist", "him", "teacher", "psychologist", "women", "black", "blacks", "white", "whites", "poor", "female", "woman", "girl", "girls", "technician", "surgeon", "chinese", "detective", "sergeant", "bourgeois", "prostitute", "offenses", "richest", "hispanic", "aggravated", "offences", "convictions", "privileged", "african", "indian", "korean", "feminine", "poorer", "latino", "needy", "therapist", "affluent", "psychiatrist", "european", "robbery", "felony", "kidnapping", "homicide", "japanese", "spanish", "mexican", "indians", "pakistani", "wealthy", "billionaire", "homeless", "murder", "instructor", "teenager", "colored", "classmate", "bartender", "crimes", "conviction", "crime", "terrorist", "militant", "terror", "terrorists", "terrorism", "jihad", "nurse", "guys", "supervisor", "caucasian", "millionaire"]
curr = ['he', 'his', 'him', 'man', 'men', 'she', 'her', 'women', 'wealthy', 'Man', 'teacher', 'teenager', 'male', 'Woman', 'black', 'Black', 'crimes', 'white', 'chemist', 'European', 'supervisor', 'teen', 'girls', 'India', 'female', 'Asian', 'poverty', 'rich', 'feminine', 'Mexico', 'boys', 'China', 'Japan', 'poor', 'Men', 'Chinese', 'Women', 'Korea', 'boy', 'billionaire', 'girl', 'arson', 'murder', 'nurse', 'underprivileged', 'therapist', 'woman', 'convictions', 'blacks', 'whites', 'privileged', 'terror', 'poorest', 'Nepal', 'surgeon', 'detective', 'crime', 'terrorism', 'Asians', 'conviction', 'bourgeois', 'militant', 'affluent', 'felony', 'colored', 'classmate', 'terrorist', 'extremism', 'Caucasian', 'Bangladesh', 'psychologist', 'guy', 'Caucasoid', 'homeless', 'psychiatrist', 'sergeant', 'guys', 'robbery', 'perjury', 'extortion', 'instructor', 'Islamist', 'poorer', 'kidnapping', 'aggravated', 'richest', 'asian', 'offenses', 'Hezbollah', 'millionaire', 'burglary', 'offences', 'terrorists', 'ISIS', 'extremist', 'penniless', 'technician', 'indian', 'bartender', 'manslaughter', 'homicide', 'radicalization', 'affluence', 'bourgeoisie', 'janitor', 'paramedic', 'misdemeanor', 'spanish', 'needy', 'extremists', 'prostitute', 'Blacks', 'DUI', 'jihad', 'african', 'firefighter', 'Afro', 'Islamists', 'Salafist', 'jihadists', 'felonies', 'advantaged', 'girly', 'jihadist', 'PKK', 'hispanic', 'japanese', 'mexican', 'european', 'latina', 'AQAP', 'chinese', 'caucasian', 'jihadi']
embedkey4 = ['he', 'his', 'him', 'man', 'men', 'she', 'her', 'women', 'wealthy', 'Man', 'teacher', 'teenager', 'male', 'Woman', 'black', 'Black', 'crimes', 'white', 'chemist', 'European', 'supervisor', 'teen', 'girls', 'India', 'female', 'Asian', 'poverty', 'rich', 'feminine', 'Mexico', 'boys', 'China', 'Japan', 'poor', 'Men', 'Chinese', 'Women', 'Korea', 'boy', 'billionaire', 'girl', 'arson', 'murder', 'nurse', 'underprivileged', 'therapist', 'woman', 'convictions', 'blacks', 'whites', 'privileged', 'terror', 'poorest', 'Nepal', 'surgeon', 'detective', 'crime', 'terrorism', 'Asians', 'conviction', 'bourgeois', 'militant', 'affluent', 'felony', 'colored', 'classmate', 'terrorist', 'extremism', 'Caucasian', 'Bangladesh', 'psychologist', 'guy', 'Caucasoid', 'homeless', 'psychiatrist', 'sergeant', 'guys', 'robbery', 'perjury', 'extortion', 'instructor', 'Islamist', 'poorer', 'kidnapping', 'aggravated', 'richest', 'asian', 'offenses', 'Hezbollah', 'millionaire', 'burglary', 'offences', 'terrorists', 'ISIS', 'extremist', 'penniless', 'technician', 'indian', 'bartender', 'manslaughter', 'homicide', 'radicalization', 'affluence', 'bourgeoisie', 'janitor', 'paramedic', 'misdemeanor', 'spanish', 'needy', 'extremists', 'prostitute', 'Blacks', 'DUI', 'jihad', 'african', 'firefighter', 'Afro', 'Islamists', 'Salafist', 'jihadists', 'felonies', 'advantaged', 'girly', 'jihadist', 'PKK', 'hispanic', 'japanese', 'mexican', 'european', 'latina', 'chinese', 'caucasian', 'jihadi']
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


diff = list(set(all_words) - set(curr))
diff2 = list(set(all_words) - set(embedkey4))
print(len(all_words))
print(len(curr))
print(len(diff), diff)
print(len(diff2), diff2)