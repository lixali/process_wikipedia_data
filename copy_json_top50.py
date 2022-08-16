import subprocess
import json
# JSON file
f = open ('containfileReduced.json', "r")
  
# Reading from file
data = json.loads(f.read())
  
# Iterating through the json
# list
subprocess.run(["mkdir", "-p", "saved_json_top50/" ])
for word in data:
    for txtfile in data[word]:
        #print(txtfile)
        currfile = txtfile
        subprocess.run(["cp",  txtfile , "saved_json_top50/" ])
        #subprocess.run(["cp",  txtfile, "../saved_json_top50/"])
  
# Closing file
f.close()
