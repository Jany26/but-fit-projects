# MongoDB part of UPA project

Versions (developed and tested on 22.04.3 LTS x86_64):
- MongoDB:		7.0.2
- Mongosh:		2.0.1
- Python:       3.10.12

```shell
docker build -t my-mongodb-image .
```
- downloads and builds an image (with mongoDB) - needs to be run in this folder (where the Dockerfile is)

```shell
docker run --name my-mongodb-container -d -p 27017:27017 my-mongodb-image
```
- runs the docker container (first time), any subsequent container starts are done with:

```shell
docker start my-mongodb-container
```

```shell
docker exec -it my-mongodb-container mongosh
```
- runs the mongo shell inside the container (for manual debugging and checking the data etc.)

After the docker image is set up, demonstration scripts can be run:

```shell
python3 load_mongo.py
```
- this script downloads the dataset (if needed) and inserts data into the database 

```shell
python3 query_mongo.py
```
- this script runs 3 queries on the database and returns results as a pandas dataframe formatted output

For cleaning up, use these commands:

```shell
python3 drop_mongo.py
```
- this script removes the database

```shell
docker stop my-mongodb-container
```
- stops the running container (with mongoDB on it)

```shell
docker rm my-mongodb-container
``` 
- removes the container

```shell
docker rmi my-mongodb-image
```
- removes the image