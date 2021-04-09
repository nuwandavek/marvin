# App

App related `README` file. 

## To initialise the sqlite database
Step1: from tables import db <br/>
Step2: db.create_all() <br/>
### NOTE: No need to create the db now as it is already instantiated

## Tables
### User: <br/>
id | username | email| password | time_stamp | documents <br/>


### Docs: <br/>
id | user_id | title | time_stamp | content  | styles | saliencies | author <br/>

### Style: <br/>
time_stamp | docid | id | intention | input | output | document

### Saliency: <br/>
time_stamp | docid | id | input | output | document








