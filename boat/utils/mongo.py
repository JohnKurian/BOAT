import pymongo
from pymongo import MongoClient
from boat.config import Config

def get_mongo_client() -> MongoClient:
    """
    Get a MongoDB client instance.
    """
    return MongoClient(Config.MONGO_URI)

def get_database(client: MongoClient):
    """
    Get the application database.
    """
    return client[Config.DB_NAME]
