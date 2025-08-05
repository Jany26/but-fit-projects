from pymongo import MongoClient

URL = "mongodb://localhost:27017/"
DB_NAME = "mydb"

DATASET = "Dopravní_nehody___Traffic_accidents.geojson"
LINK = f"https://data.brno.cz/datasets/mestobrno::dopravn%C3%AD-nehody-traffic-accidents.geojson?where=1=1&outSR=%7B%22latestWkid%22%3A5514%2C%22wkid%22%3A102067%7D"


class ClientDriver:
    def __init__(self, url, db_name):
        self.client: MongoClient = MongoClient(url)
        self.db_name: str = db_name
        self.url: str = url
        self.db = self.client[self.db_name]

    def drop_database(self):
        self.client.drop_database(self.db_name)

    def get_collection_count(self, collection: str):
        return self.db[collection].count_documents({})

    def drop_collection(self, collection: str):
        self.db[collection].drop()

    def close(self):
        self.client.close()
