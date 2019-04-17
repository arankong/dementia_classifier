# dementia_classifier
Code for my masters thesis. The baseline methods were implemented by Vaden Masrani (https://github.com/vmasrani/dementia_classifier), a current PhD student at UBC. To run: 
- Request access to the [DementiaBank dataset](https://dementia.talkbank.org) from either Davida Fromm: fromm@andrew.cmu.edu or Brian MacWhinney: macw@cmu.edu  
- Once you have permission, email me at kongw@alumni.ubc.ca to get a copy of the preprocessed dataset.
    -  Place data alongside run.py.
    -  Place lib within dementia_classifier/ 
- Configure Python virtual environment. Use pip to install virtualenv:
    ```bash
    pip install virtualenv
    ```
    And start the Python virtual environment using command:
    ```bash
    source ./dp-env/bin/activate
    ```        
    If you try running the project, there will be errors saying unable to find some libraries. In the virtual environment, install those libraries using pip:
    ```bash
    pip install module_name
    ```     

- Note: May need to install the 'stopwords' and 'punkt' package for the NLTK python package. 
- Make sure sql is installed on your system and create a database with the appropriate permissions to store the processed data and results. Modify dementia_classifier/db.py with the appropriate user, password, and database name. If you don’t want to change the code in db.py, you need to create a MySQL user named 'patata' with a password 'carota'. Then create a database named 'dementia_data' and grant all privileges to the 'patata' user.
    ```sql
    mysql> GRANT ALL PRIVILEGES ON dementia_data.* TO 'patata'@'localhost';
    ```

- Start the stanford parser with:
    ```bash
    java -Xmx4g -cp "dementia_classifier/lib/stanford/stanford-corenlp-full-2015-12-09/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 20000
    ```
- In the virtual environment, run with:
  ```bash
  python run.py
  ```


## Troubleshooting
You would probably get “ImportError: cannot import name check_build” even if you have already installed sklearn. To solve the problem, try installing scipy (may need to restart the python shell after installing scipy).

Error: "RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
"
See: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

Error: "ValueError: invalid literal for int() with base 10: 'sh: dementia_classifier/lib/SCA/L2SCA/./tregex.sh: Permission denied'". Check permissions for tregex.sh:
```bash
chmod 755 dementia_classifier/lib/SCA/L2SCA/tregex.sh
```