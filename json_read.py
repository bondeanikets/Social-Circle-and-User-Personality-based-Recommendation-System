import json
import sys

def getMaxFriendUserJson(file_path, record_users):
    maxFriends = 0;
    maxFriendsJson = '';
    with open(file_path) as f:
      for line in f:
        jo = json.loads(line);
        record_users[jo['user_id']] = '';
        numFriends = len(jo['friends']);
        if ( numFriends > maxFriends ):
          maxFriends = numFriends;
          maxFriendsJson = jo;
    return maxFriendsJson;

def populateAllUsers(filtered_users, numLevels, record_users):
    for level in range(numLevels):
      with open(file_path) as f:
        for line in f:
          jo = json.loads(line);
          user = jo['user_id'];
          if (filtered_users.has_key(user)):
            for entry in jo['friends']:
              if (record_users.has_key(entry)):
                filtered_users[entry] = '';
      print level+2,":#users", len(filtered_users);

def writeFile(file_path, comp_file_path, filtered_users):
    fw = open(comp_file_path, "w");
    with open(file_path) as f:
      for line in f:
        jo = json.loads(line);
        user=jo['user_id'];
        if (not filtered_users.has_key(user)):
            continue;
        friends=jo['friends'];
        new_friends = [];
        for friend in friends:
          if (filtered_users.has_key(friend)):
            new_friends.append(friend);
        jo['friends'] = new_friends;
        fw.write(json.dumps(jo,indent = 4));
    fw.close();

file_path = 'yelp_academic_dataset_user.json'; 
comp_file_path = 'comp_yelp_academic_dataset_user.json'; 
numLevelsMinusOne = 1;
record_users = {};
maxFriendsJson = getMaxFriendUserJson(file_path, record_users);
filtered_users = {};
filtered_users[maxFriendsJson['user_id']] = '';
for entry in maxFriendsJson['friends']:
  if (record_users.has_key(entry)):
    filtered_users[entry] = '';
print 1,":#users", len(filtered_users);
populateAllUsers(filtered_users, numLevelsMinusOne, record_users);
writeFile(file_path, comp_file_path, filtered_users);
