import os
import pymongo
from bson.json_util import dumps
import json
import gridfs




client = pymongo.MongoClient(
     "mongodb+srv://root:7dc41992@cluster0.vckj9.mongodb.net/main?retryWrites=true&w=majority")


pipeline = [
   {
     '$match': { 'fullDocument.exp_id': "7f8608e2-14c6-4071-864c-9ae5306324fd" }
   } ]


# runs = []
#
#
# for run in client.automl.runs.find({}):
#      run = json.loads(dumps(run))
#      runs.append(json.loads(dumps(run)))
#
# print(runs)
#
# change_stream = client.automl.runs.watch(pipeline)
# for change in change_stream:
#     print(json.loads(dumps(change))['fullDocument'])
#     print('') # for readability only

#
# best_config = None
# for run in client.automl.runs.find({'exp_id': { '$eq': '7f8608e2-14c6-4071-864c-9ae5306324fd' } }).sort([("r2", pymongo.DESCENDING)]):
#     best_config = run
#     break
#
# print(best_config)

# fs = gridfs.GridFSBucket(client.automl)
# fs.get('5ffe4c4b0bff9f6f25f1b0c1')
# print(fs.)

# file_id = fs.upload_from_stream("trials_obj", open( r'debug.log', 'rb'))
# print(file_id)
#
#
# file = open('myfile','wb+')
# fs.download_to_stream(file_id, file)
# fileID = fs.put( open( r'debug.log', 'rb')  )

# print(fileID)



# client.automl.experiments.update({'exp_id': {'$eq': '7f8608e2-14c6-4071-864c-9ae5306324fd'}}, {'$set': {'trials_file_id': '0'}})


# print(list(client.automl.experiments.find({'exp_id': { '$eq': '7f8608e2-14c6-4071-864c-9ae5306324fd' } }))[0])

print('\n')
for result in list(client.automl.runs.find({'exp_id': { '$eq': 'ae509fee-d082-488c-b7d8-4bf82475c2ea' } }, {'r2': 1, 'rmse': 1, 'config': 1}).sort([("r2", pymongo.DESCENDING)])):
    print(result)
    print('\n')



# print(list(client.automl.experiments.find({ })))

