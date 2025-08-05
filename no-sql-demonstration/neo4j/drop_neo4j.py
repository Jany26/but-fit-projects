from utils_neo4j import *

if __name__ == "__main__":
    driver = Neo4jDriver(URI, USERNAME, PASSWORD)
    driver.drop_database_in_batches(20000)
    driver.close()
