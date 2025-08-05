from pymongo import GEOSPHERE
import json
import os
import urllib.request

from utils_mongo import *


def get_needed_fields(json_path: str) -> dict:
    with (open(json_path) as f):
        data = json.load(f)
        needed_fields = {k: True for k in data["features"][0]["properties"].keys()}
        for accident in data["features"]:
            for property, value in accident["properties"].items():
                if value is None:
                    needed_fields[property] = False
        for prop, needed in needed_fields.items():
            print(f"{prop :<40} {'needed' if needed else 'opt'}")
    return needed_fields


def cleanup_pedestrian_data(accident: dict) -> dict:
    pedestrian_data = [
        "kategorie_chodce",
        "stav_chodce",
        "chovani_chodce",
        "situace_nehody",
        "prvni_pomoc",
        "nasledky_chodce",
    ]
    pedestrian_nulls = [
        accident[pedestrian_data[0]] is None,
        accident[pedestrian_data[1]] is None,
        accident[pedestrian_data[2]] is None,
        accident[pedestrian_data[3]] is None,
        accident[pedestrian_data[4]] is None,
        accident[pedestrian_data[5]] is None,
    ]

    pedestrian_involvement = None
    if not all(pedestrian_nulls):
        pedestrian_involvement = {key: accident[key] for key in pedestrian_data}
    accident["ucast_chodce"] = pedestrian_involvement
    for key in pedestrian_data:
        accident.pop(key, None)
    return accident


def cleanup_time_data(accident: dict) -> dict:
    redundant_time_data = [
        "cas",
        "hodina",
        "doba",
        "den",
        "den_v_tydnu",
        "mesic_t",
        "mesic",
        "rok",
    ]

    time = str(accident["cas"]).zfill(4)
    if int(time[:2]) > 23 or int(time[2:]) > 59:  # invalid time // usually 2560
        accident["datum"] = accident["datum"].replace("T00:00:00Z", "")
    else:
        accident["datum"] = accident["datum"].replace(
            "T00:00:00Z", f"T{time[:2]}:{time[2:]}:00+00:00"
        )
    for key in redundant_time_data:
        accident.pop(key, None)
    return accident


def remove_obsolete_data(accident: dict) -> dict:
    attributes_to_remove = [
        "x",
        "y",
        "d",
        "e",
        "id",
        "id_nehody",
    ]
    for key in attributes_to_remove:
        accident.pop(key, None)
    return accident


def create_accident_struct(data: dict) -> dict:
    accident: dict = data["properties"]
    accident["_id"] = accident.pop("OBJECTID")
    accident = cleanup_pedestrian_data(accident)
    accident = cleanup_time_data(accident)
    accident = remove_obsolete_data(accident)

    accident["location"] = data["geometry"]
    return accident


def create_collection_from_json(json_path: str, collection):
    print("creating collection from json ...")
    with open(json_path) as f:
        data = json.load(f)
        for i in data["features"]:
            accident = create_accident_struct(i)
            collection.update_one(
                {"_id": accident["_id"]},
                {"$set": accident},
                upsert=True
            )


def get_dataset(url, filename):
    if not os.path.isfile(filename):
        print("downloading dataset ...")
        urllib.request.urlretrieve(url, filename)


if __name__ == "__main__":
    client = ClientDriver(URL, DB_NAME)
    car_accident_collection = client.db["car_accidents"]

    get_dataset(LINK, DATASET)
    create_collection_from_json(DATASET, car_accident_collection)

    car_accident_collection.create_index([("location", GEOSPHERE)])
    client.close()
