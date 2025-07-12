from pymongo import MongoClient
from bson.objectid import ObjectId
from config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client.smartpantry

def get_items_for_user(email):
    return list(db.items.find({"user_email": email}))

# READ Operations
def get_items():
    return list(db.items.find())

def get_item_by_id(item_id):
    return db.items.find_one({"_id": ObjectId(item_id)})

def get_alerts():
    return list(db.alerts.find())

def get_profiles():
    return list(db.profiles.find())

def get_recipes():
    return list(db.recipes.find())

# INSERT Operations
def insert_item(item_data):
    return db.items.insert_one(item_data)

def insert_alert(alert_data):
    return db.alerts.insert_one(alert_data)

def insert_profile(profile_data):
    return db.profiles.insert_one(profile_data)

def insert_recipe(recipe_data):
    return db.recipes.insert_one(recipe_data)

def get_user_by_email(email):
    return db.users.find_one({"email": email})

def insert_user(user_data):
    db.users.insert_one(user_data)

# UPDATE Operation
def update_item(item_id, update_data):
    db.items.update_one({"_id": ObjectId(item_id)}, {"$set": update_data})

# DELETE Operation
def delete_item_by_id(item_id):
    db.items.delete_one({"_id": ObjectId(item_id)})
