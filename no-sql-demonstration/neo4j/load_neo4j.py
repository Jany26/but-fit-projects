from utils_neo4j import *
import json


def clean_house_data(data):
    data_str = (
        json.dumps(data)
        .replace('" "', "null")
        .replace('"<Null>"', "null")
        .replace('"kod_skoly": 1,', '"kod_skoly": 0,')
        .replace('"kod_skoly": 2,', '"kod_skoly": 0,')
        .replace('"kod_skoly": 3,', '"kod_skoly": 0,')
        .replace('"kod_skoly": 22117,', '"kod_skoly": 2217,')
    )
    data = json.loads(data_str)

    return data


def clean_school_data(data):
    data_str = json.dumps(data).replace('" "', "null")
    data = json.loads(data_str)
    return data


def define_constraints(driver):
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:School) REQUIRE (s.id) IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:House) REQUIRE (h.id) IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Municipality) REQUIRE (m.id) IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TownPart) REQUIRE (t.id) IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cadastre) REQUIRE (c.id) IS UNIQUE"
        )

        session.run("CREATE INDEX IF NOT EXISTS FOR (s:School) ON (s.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (h:House) ON (h.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (m:Municipality) ON (m.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:TownPart) ON (t.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Cadastre) ON (c.id)")


def load_school_data(driver, json_path):
    with open(json_path) as f:
        with driver.session() as session:
            data = json.load(f)
            data = clean_school_data(data)
            with session.begin_transaction() as tx:
                for feature in data["features"]:
                    insert_school_data(tx, feature)


def insert_school_data(tx, feature):
    school = feature["properties"]
    longitude, latitude = feature["geometry"]["coordinates"]

    insert_school_query = f"""
    MERGE (s:School {{
        id: $id,
        name: $name,
        street: COALESCE($street, ''),
        conscription_no: $conscription_no,
        orientation_no: $orientation_no,
        orientation_letter: COALESCE($orientation_letter, ''),
        postcode: $postcode,
        municipality: COALESCE($municipality, ''),
        x: $longitude,
        y: $latitude
    }})
    """

    parameters = {
        "id": school["kod"],  # houses reference the school by this code
        "name": school["nazev_cely"],  # type of school
        "street": school["ulice_nazev"],
        "orientation_no": school["cislo_orientacni_cislo"],
        "orientation_letter": school["cislo_orientacni_hodnota"],
        "conscription_no": school["cislo_domovni"],
        "postcode": school["adrp_psc"],
        "municipality": school["naz_cobce"],  # references Municipality node
        "longitude": longitude,
        "latitude": latitude,
    }
    tx.run(insert_school_query, parameters)


def load_house_data(driver, json_path):
    with open(json_path) as f:
        with driver.session() as session:
            data = json.load(f)
            data = clean_house_data(data)
            with session.begin_transaction() as tx:
                for feature in data["features"]:
                    insert_house_data(tx, feature)


def insert_house_data(tx, feature):
    house = feature["properties"]
    longitude, latitude = feature["geometry"]["coordinates"]
    insert_house_query = f"""
    MERGE (h:House {{ 
        id: $id,
        code: $code,
        address: COALESCE($address, ''),
        street: COALESCE($street, ''),
        conscription_no: $conscription_no,
        orientation_no: $orientation_no,
        orientation_letter: COALESCE($orientation_letter, ''),
        postcode: $postcode,
        x: $longitude,
        y: $latitude
    }})

    WITH h WHERE $school_id <> 0
    MERGE (s:School {{id: $school_id}})
    WITH h, s MERGE (h)-[:IN_CATCHMENT_AREA]->(s)

    WITH h WHERE $town_part_id <> 0
    MERGE (t:TownPart {{id: $town_part_id, name: COALESCE($town_part_name, '')}})
    WITH h, t MERGE (h)-[:IN_TOWN_PART]->(t)

    WITH h WHERE $municipality_id <> 0
    MERGE (m:Municipality {{id: $municipality_id, name: COALESCE($municipality_name, '')}})
    WITH h, m MERGE (h)-[:IN_MUNICIPALITY]->(m)
    
    WITH h WHERE $cadastre_id <> 0
    MERGE (c:Cadastre {{id: $cadastre_id, name: COALESCE($cadastre_name, '')}})
    MERGE (h)-[:IN_CADASTRE]->(c)
    """

    parameters = {
        "id": house["objectid"],
        "code": house["kod"],
        "conscription_no": house["cislo_domo"],
        "orientation_no": house["cislo_orie"],
        "orientation_letter": house["cislo_or_1"],
        "postcode": house["psc"],
        "street": house["ulice_naze"],
        "address": house["adresa"],
        "municipality_id": house["cobce_kod"],  # Municipality node reference
        "municipality_name": house["cobce_naze"],
        "cadastre_id": house["katuze_kod"],  # Cadastre node reference
        "cadastre_name": house["katuze_naz"],
        "town_part_id": house["momc_kod"],  # TownPart node reference
        "town_part_name": house["momc_nazev"],
        "school_id": house["kod_skoly"],  # School node reference
        "longitude": longitude,
        "latitude": latitude,
    }
    tx.run(insert_house_query, parameters)


if __name__ == "__main__":
    driver: neo4j.GraphDatabase.driver = Neo4jDriver(URI, USERNAME, PASSWORD)
    define_constraints(driver.driver)
    get_dataset(JSON_SCHOOLS_URL, DATASET_SCHOOLS)
    get_dataset(JSON_HOUSES_URL, DATASET_HOUSES)
    load_school_data(driver.driver, DATASET_SCHOOLS)
    load_house_data(driver.driver, DATASET_HOUSES)
    driver.close()
