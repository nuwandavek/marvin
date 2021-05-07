# App


## Running the db
```
bash run_mysql_docker.sh
```
If this is the first time running it, you've to create the table
```
docker exec -it test-mysql bash
mysql -uroot -p
# The password is root
USE marvin;
CREATE TABLE accepted_transfers (
    id int NOT NULL AUTO_INCREMENT,     
    mode varchar(25) NOT NULL,
    goal varchar(50) not null, 
    original varchar(300) not null,
    original_val varchar(50) not null,
    accepted varchar(300) not null, 
    accepted_val varchar(50) not null, 
    PRIMARY KEY (id) 
);

#To check the table
select * from accepted_transfers limit 10;
```

## Running the App
```
#Without Docker
python server.py

#With Docker
bash run_app_docker.sh
```

## Build the docker images
```
bash ../build_docker_images.sh
```