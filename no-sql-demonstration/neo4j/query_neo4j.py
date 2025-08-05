from utils_neo4j import *
import json


# which 5 schools have the biggest catchment area (most houses attached to them)
def query1(driver):
    query = f"""
    MATCH (s:School)
    OPTIONAL MATCH (h:House)-[:IN_CATCHMENT_AREA]->(s)
    WITH s, COUNT(h) AS catchmentArea
    ORDER BY catchmentArea DESC
    LIMIT 5 RETURN s, catchmentArea
    """
    with driver.session() as session:
        query_result = session.run(query)
        data = []
        print("QUERY 1: SCHOOLS WITH BIGGEST CATCHMENT AREA")
        for record in query_result:
            school = record["s"]._properties
            street = f", {school['street']}" if school["street"] != "" else " "
            data.append(
                {
                    "id": school["id"],
                    "name": school["name"],
                    "address": f"{school['municipality']}{street} {school['orientation_no']}{school['orientation_letter']}",
                    "house_count": record["catchmentArea"],
                }
            )
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        print(json_data)


# which municipality has the most schools
def query2(driver):
    query = f"""
    MATCH (h:House)-[:IN_CATCHMENT_AREA]->(s:School)
    MATCH (h)-[:IN_MUNICIPALITY]->(m:Municipality)
    WITH m, COLLECT(DISTINCT s) AS schools
    RETURN m, SIZE(schools) AS numSchools
    ORDER BY numSchools DESC
    LIMIT 5
    """
    with driver.session() as session:
        query_result = session.run(query)
        data = []
        print("QUERY 2: MUNICIPALITIES WITH MOST SCHOOLS")
        for record in query_result:
            municipality = record["m"]._properties
            data.append(
                {
                    "id": municipality["id"],
                    "name": municipality["name"],
                    "school_count": record["numSchools"],
                }
            )
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        print(json_data)


def analyse_MN_relations(json_path, code1, code2):
    with open(json_path) as f:
        relation = {}
        data = json.load(f)
        for feature in data["features"]:
            house = feature["properties"]
            val1 = house[code1]
            valN = house[code2]
            if val1 != 0:
                if val1 not in relation:
                    relation[val1] = set()
                if valN != 0:
                    relation[val1].add(valN)
    return relation


if __name__ == "__main__":
    driver = Neo4jDriver(URI, USERNAME, PASSWORD)
    query1(driver.driver)
    query2(driver.driver)
    driver.close()
