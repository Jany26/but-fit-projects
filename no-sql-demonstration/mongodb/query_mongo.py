from utils_mongo import *
import pandas as pd


# show the number accidents in an area 250m from the centre by weekday
def mongodb_query_1(collection):
    geopoint_namesti_svobody = {"type": "Point", "coordinates": [16.6084, 49.1951]}

    day_mapping = {
        2: "pondělí",
        3: "úterý",
        4: "středa",
        5: "čtvrtek",
        6: "pátek",
        7: "sobota",
        1: "neděle",
    }

    city_centre_by_weekday = [
        {
            "$geoNear": {
                "near": geopoint_namesti_svobody,
                "distanceField": "distance",
                "maxDistance": 250,
                "spherical": True,
            }
        },
        {
            "$addFields": {
                "year": {"$year": {"$dateFromString": {"dateString": "$datum"}}},
                "week_day": {
                    "$dayOfWeek": {"$dateFromString": {"dateString": "$datum"}}
                },
            }
        },
        {"$match": {"year": 2022}},
        {"$group": {"_id": "$week_day", "total": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    query1 = pd.DataFrame(collection.aggregate(city_centre_by_weekday))
    query1 = query1.assign(_id=query1._id.map(day_mapping))
    print("QUERY 1: ACCIDENTS IN THE CENTRE GROUPED BY WEEKDAY")
    print(query1)
    print()


# show accidents involving pedestrians grouped and sorted into ranges by material damage
def mongodb_query_2(collection):
    buckets = [
        0,
        100,
        500,
        1_000,
        2_000,
        3_000,
        4_000,
        5_000,
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
        60_000,
        70_000,
        80_000,
        90_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        1_000_000,
    ]
    damage_mapping = {val: f"{val}+" for val in buckets}
    damage_mapping["1000000+"] = "1000000+"

    pedestrian_involvement = [
        {"$match": {"ucast_chodce": {"$ne": None}}},
        {
            "$bucket": {
                "groupBy": "$hmotna_skoda",
                "boundaries": buckets,
                "default": "1000000+",
                "output": {"count": {"$sum": 1}},
            }
        },
        {"$sort": {"_id": -1}},
    ]
    query2 = pd.DataFrame(collection.aggregate(pedestrian_involvement))
    query2 = query2.assign(_id=query2._id.map(damage_mapping))
    print("QUERY 2: ACCIDENTS INVOLVING PEDESTRIANS GROUPED BY MATERIAL DAMAGE")
    print(query2)
    print()


# show accidents that were caused by alcohol usage and happened during
# Christmas (i.e. from Dec 23rd to 27th)
def mongodb_query_3(collection):
    alcohol_during_christmas = [
        {
            "$addFields": {
                "month": {"$month": {"$dateFromString": {"dateString": "$datum"}}},
                "day_of_month": {
                    "$dayOfMonth": {"$dateFromString": {"dateString": "$datum"}}
                },
            }
        },
        {
            "$match": {
                "alkohol_vinik": "ano",
                "month": 12,
                "day_of_month": {"$gte": 23, "$lte": 27},
            }
        },
        {"$project": {"datum": 1, "hmotna_skoda": 1, "_id": 1}},
        {"$sort": {"datum": -1}},
    ]
    query3 = pd.DataFrame(collection.aggregate(alcohol_during_christmas))
    print("QUERY 3: ACCIDENTS INVOLVING ALCOHOL THAT HAPPENED DURING CHRISTMAS")
    print(query3)
    print()


def count_unique(client):
    counter = 0
    cursor = client.db["car_accidents"].find({})
    unique_accidents = set()
    for i in cursor:
        x = i["location"]["coordinates"][0]
        y = i["location"]["coordinates"][1]
        datetime = i["datum"]
        unique_accidents.add((x, y, datetime))
        counter += 1
    print(counter)
    print(len(unique_accidents))


if __name__ == "__main__":
    client = ClientDriver(URL, DB_NAME)
    car_accident_collection = client.db["car_accidents"]
    mongodb_query_1(car_accident_collection)
    mongodb_query_2(car_accident_collection)
    mongodb_query_3(car_accident_collection)
    client.close()
