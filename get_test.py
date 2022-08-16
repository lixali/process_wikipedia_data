import json
file = "./mytest.json"
with open(file,"a",encoding='utf-8') as currfile:
    json_object = json.dumps({"mykey": ["sdfsdf", "sdfsdf"]}, indent = 4)
    currfile.write(json_object)
    json_object = json.dumps({"mykey2": ["234234", "234234"]}, indent = 4)
    currfile.write(json_object)



