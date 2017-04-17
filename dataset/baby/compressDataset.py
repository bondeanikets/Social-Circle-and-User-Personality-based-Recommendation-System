import json
import sys
import operator
import random

def findAllUsers(file_path_user, record_users):
    with open(file_path_user) as f:
      for line in f:
        jo = json.loads(line);
        record_users[jo['user_id']] = 1;
    return;

def genRandomNumbers(n, high):
    random.seed(1);
    picks = {};
    for i in range(n):
        num = random.randrange(1, high);
        picks[num] = 1;
    return picks;

def getBaseUsers(file_path_user, rh):
    l= 1;
    bu = {};
    with open(file_path_user) as f:
      for line in f:
        jo = json.loads(line);
        if (rh.has_key(l)): 
          bu[jo['user_id']] = 1;
        l+= 1;
    return bu

def expandBaseToOneLevel(file_path_user, baseUsers, record_users):
    users = {};
    with open(file_path_user) as f:
      for line in f:
        jo = json.loads(line);
        user = jo['user_id'];
        if baseUsers.has_key(user):
            users[user] = 1;
            for f in jo['friends']:
              if record_users.has_key(f):
                  users[f] = 1;
    return users;

def writeUserFile(file_path_user, comp_file_path_user, filtered_users):
    fw = open(comp_file_path_user, "w");
    f1 = 0; 
    with open(file_path_user) as f:
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
        if (len(new_friends) == 1):
            f1 += 1;
        #fw.write(json.dump(jo, indent=4));
        json.dump(jo, fw);
        fw.write("\n");
    fw.close();
    print "1friends:", f1

def writeReviewFile(file_path_review, comp_file_path_review, filtered_users, businesses):
    fw = open(comp_file_path_review, "w");
    f1 = 0; 
    with open(file_path_review) as f:
      for line in f:
        jo = json.loads(line);
        user=jo['user_id'];
        if (not filtered_users.has_key(user)):
            continue;
        b = jo['business_id'];        
        businesses[b] = 1;
        json.dump(jo, fw);
        fw.write("\n");
    fw.close();

def writeReviewFile(file_path_review, comp_file_path_review, filtered_users, businesses):
    fw = open(comp_file_path_review, "w");
    f1 = 0; 
    with open(file_path_review) as f:
      for line in f:
        jo = json.loads(line);
        user=jo['user_id'];
        if (not filtered_users.has_key(user)):
            continue;
        b = jo['business_id'];        
        businesses[b] = 1;
        json.dump(jo, fw);
        fw.write("\n");
    fw.close();

def writeBusinessFile(file_path_user, comp_file_path_user, filtered_users, businesses):
    fw = open(comp_file_path_business, "w");
    f1 = 0; 
    with open(file_path_business) as f:
      for line in f:
        jo = json.loads(line);
        business=jo['business_id'];
        if (not businesses.has_key(business)):
            continue;
        json.dump(jo, fw);
        fw.write("\n");
    fw.close();

file_path_user = 'yelp_academic_dataset_user.json'; 
file_path_review = 'yelp_academic_dataset_review.json'; 
file_path_business = 'yelp_academic_dataset_business.json'; 

comp_file_path_user = 'comp_yelp_academic_dataset_user.json'; 
comp_file_path_review = 'comp_yelp_academic_dataset_review.json'; 
comp_file_path_business = 'comp_yelp_academic_dataset_business.json'; 

record_users = {};
businesses = {};
findAllUsers(file_path_user, record_users);
rh = genRandomNumbers(10000, len(record_users));
baseUsers = getBaseUsers(file_path_user, rh);
users = expandBaseToOneLevel(file_path_user, baseUsers, record_users);
print len(users)
writeUserFile(file_path_user, comp_file_path_user, users);
writeReviewFile(file_path_review, comp_file_path_review, users, businesses);
print len(businesses)
writeBusinessFile(file_path_business, comp_file_path_business, users, businesses);
