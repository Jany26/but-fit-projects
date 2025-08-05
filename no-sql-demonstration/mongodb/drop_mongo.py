from utils_mongo import *

if __name__ == "__main__":
    client = ClientDriver(URL, DB_NAME)
    client.drop_collection("car_accidents")
    client.drop_database()
    client.close()
