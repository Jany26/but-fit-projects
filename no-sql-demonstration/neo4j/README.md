# Neo4j part of UPA project

Versions (developed and tested on Ubuntu 22.04.3 LTS x86_64):
- Python        3.10.12
- Neo4j:        5.12.0
- Cypher-Shell: 5.12.0

```shell
docker build -t my-neo4j-image .
```
- builds a docker image with neo4j installed - needs to be run in this folder (with Dockerfile in it)

```shell
docker run --name my-neo4j-container -p7474:7474 -p7687:7687 \
-d -v $HOME/neo4j/data:/data \
-v $HOME/neo4j/logs:/logs \
-v $HOME/neo4j/import:/var/lib/neo4j/import \
-v $HOME/neo4j/plugins:/plugins \
--env NEO4J_AUTH=neo4j/password \
my-neo4j-image
```
- runs a container with correct settings and ports, when starting the container for the first time, otherwise use:

```shell
docker start my-neo4j-container
```

```shell
docker exec -it my-neo4j-container cypher-shell
```
- starts a cypher shell inside the container (sometimes needed for debugging purposes), username: `neo4j`, password: `password`

Note: Sometimes a container will not start, because some process is occupying the port 7687. To find out the pid of the process, try running one of the following:
```shell
sudo ss -lptn "sport = :7687"
```
```shell
sudo ss -lptn "sport = :7474"
```
and kill the process occupying the needed port to start a container with the neo4j database.

After the docker image is set up, demonstration scripts can be run:

```shell
python3 load_neo4j.py
```
- this script downloads the dataset (if needed) and inserts data into the database 

```shell
python3 query_neo4j.py
```
- this script runs 2 queries on the database and returns results in a JSON-like format

```shell
python3 drop_neo4j.py
```
- this script removes the database (or rather the data in it)

```shell
docker stop my-neo4j-container
```
- stops the running container

```shell
docker rm my-neo4j-container
``` 
- removes the container

```shell
docker rmi my-neo4j-image
```
- removes the image