# Marvin: Stylistic Language Editor

## Background

At the School of Information we find ourselves at the intersection of cutting edge research and a will to democratize access to such research. Throughout history, the written word has played a vital role in expanding the horizons of human consciousness. And effective use of stylistic language is ubiquitous in this pursuit, from scientific text and Wikipedia to the writings of Shakespeare and Mahatma Gandhi. Unfortunately, language famously suffers from issues of bias, accessibility and quality control. Anyone who has ever made a Wikipedia edit or has written a technical blog post immediately recollects the dozens of examples one has to pore over, to emulate the style, objectivity and tenor of the writing style. Adherence to a particular style can often be used as a proxy measure of value. At best, this leads to a high variance in quality, and at worst, it makes the process of writing entirely prohibitive to newcomers. We believe that a system, such as Marvin, will help users generate style-conformant content and can democratize access to domains for fledgling writers from various backgrounds.

## What is Marvin?

We have created an AI powered tool for stylistic language editing called Marvin. Marvin leverages a combination of recent Natural Language Processing (NLP) and deep learning innovations in text classification and generation as well as visualization and interpretation techniques. Marvin provides the ability to classify the style of texts, transfer to other styles as well as understand how certain features contribute to style. It strives to adhere to a machine-in-the-loop framework, where writing is performed largely by human users, but is aided by algorithmic suggestions. 
We evaluated the machine learning models that comprise Marvin on several benchmark metrics and determined that these models are able to perform well on linguistic style tasks. One of Marvin’s primary novel contributions is the ability to transfer to different specific levels of a particular style. It also can perform joint style analysis and transfer for several dimensions of style simultaneously. 

**Don’t Panic!**

## Features
- Classification of text along different style axes 		
- Including macro-styles like domains and micro-styles like emotions	
- Transfer of text from one style to another 		
   - Eg: Modern English →Shakespearean
- Transfer of text from one point of a style spectrum to another 		
  - Eg: Mid Formality → High Formality
- Transfer of text to a combination of multiple styles 		
  - Eg: Low Formality and Happy →  High Formality and Sad
- Visualization of feature importance for styles 		
  - Understand how each word contributes to scores
- Record user actions and generate new datasets forming a closed machine-in-the-loop system
 
## Backend API’s:

### App Server
- /analysis [GET]: To get the style statistics of the text
- /transfer  [GET]: To get suggestions for transfer
- /transfer_action [POST]: To record the user selection to a Database
- /swap_models [POST] : To swap out the models for different transfer modes

### ML Server
- /classification [GET] : To compute the classification score
- /transfer [GET] : To get the suggestions for transfer. 
  - Here is where we also make an external API call (OpenAI GPT-3)
  - This also calls the classification API to filter and rank results
- /swap_models [POST] : To swap out the models for different transfer modes

### API Reference:
[ml/ml_server.py classification](https://github.com/nuwandavek/marvin/blob/master/ml/ml_server.py#L221)
```
@app.route('/classification', methods = ['GET'])
def get_joint_classify_and_salience():
    '''
    Inputs:
    Input is assumed to be json of the form 
      {text: "some text"}.
  
    Results:
      Run ML classification model on text. 
      
    Returns:
      res: a dict containing information on 
        classification and input salience weights.
        It has a key 'tokens' which is an array of the 
        tokenized input text. It also has a key for each 
        classification task. Each of these are themselves
        dicts containing keys for the predicted class, 
        the probability of this class, and also the salience score
        for each token from the tokenized input.
    '''
```
[app/server.py analyze](https://github.com/nuwandavek/marvin/blob/master/app/server.py#L41)
```
@app.route('/analyze', methods = ['GET'])
def get_stats():
	'''
	Inputs:
	Input is assumed to be json of the form 
  	{text: "some text"}. 
  	
  	Returns:
	Following statistics for the text :
	- Class, class probs
	- Salience over tokens for each class
  	'''
```

## Examples:

-  **Input :** He made a new therapeutic technique called logotherapy.  
   **Goal :** Convert to Shakespeare, Wikipedia, Abstract  
   **Prediction :**  
      - **Shakespeare :** He hath made a new therapeutic technique, logotherapy.  
        **Wikipedia :** He created a new therapeutic technique called logotherapy.  
        **Abstract :** He developed a novel therapeutic technique called logotherapy.  

-  **Input :** No, it’s just a silly old skit from SNL.
   **Goal :** Convert to Formal
   **Prediction :** No it’s just an old skit of SNL.

-  **Input :** No wonder the sport felt good at that Wimbledon champion’s dinner.
   **Goal :** Convert to Formal
   **Prediction :** It’s not surprising that the sport felt good at the dinner ofWimbledon champions.

-  **Input :** why do u have to ask such moronic questions?
   **Goal :** Convert to Formal
   **Prediction :** why do we have to ask such questions?

-  **Input :** Breaking Bad won all the awards at Emmy.
   **Goal :** Convert to Wikipedia
   **Prediction :** Breaking Bad won all the Emmy Awards for its outstanding performance.

-  **Input :** What do you think?  
   **Goal :** Convert to Shakespeare - high  
   **Prediction :** What dost thou say?  

## Running Marvin:
Marvin has two server - docker images (App, ) and a MySQL docker image, .
Instructions to run the servers are in the README.md files of the app and ml folders respectively. 

