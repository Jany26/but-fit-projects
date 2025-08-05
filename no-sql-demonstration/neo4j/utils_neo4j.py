import neo4j
import os
import urllib.request


URI = "neo4j://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password"

DATASET_SCHOOLS = "Základní_školy_-_seznam___Primary_schools_-_list.geojson"
DATASET_HOUSES = (
    "Základní_školy_-_spádové_oblasti___Primary_schools_-_catchment_areas.geojson"
)

JSON_SCHOOLS_URL = f"https://data.brno.cz/datasets/mestobrno::z%C3%A1kladn%C3%AD-%C5%A1koly-seznam-primary-schools-list.geojson?where=1=1&outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D"
JSON_HOUSES_URL = f"https://data.brno.cz/datasets/mestobrno::z%C3%A1kladn%C3%AD-%C5%A1koly-sp%C3%A1dov%C3%A9-oblasti-primary-schools-catchment-areas.geojson?where=1=1&outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D"


def get_dataset(url, filename):
    if not os.path.isfile(filename):
        print("downloading dataset ...")
        urllib.request.urlretrieve(url, filename)


class Neo4jDriver:
    def __init__(self, uri, username, password):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def drop_database(self):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run(
                    """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                DELETE n,r;
                """
                )

    def drop_database_in_batches(self, batch_size):
        driver = self.driver
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            total_count = result.single()["count"]
            num_batches = total_count // batch_size + (
                1 if total_count % batch_size > 0 else 0
            )
            for batch in range(num_batches):
                offset = batch * batch_size
                with session.begin_transaction() as tx:
                    tx.run(
                        "MATCH (n)-[r]-() WITH n, r LIMIT $batch_size DELETE r",
                        batch_size=batch_size,
                    )
                    tx.run(
                        "MATCH (n) WITH n LIMIT $batch_size DETACH DELETE n",
                        batch_size=batch_size,
                    )
        driver.close()
