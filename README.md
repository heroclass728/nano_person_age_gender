# Overview

This project is to detect person, his face, age and gender through web camera. The real environment is person's taking 
into the bus. When person take into the bus, the Jetson Nano connected with web camera counts the number of person, 
estimate the age and gender of each person, and then send these information to a server.

## Project Structure
The main content is in src folder

- age_gender
    
    source to detect human face and estimate his age and gender.

- info_commun
    
    source to send all of the extracted information to the server

- person

    source to detect the person taking into the bus and count them

- utils

    source with folder_file management functionality and models used in machine learning

- app

    the main execution source
    
- settings

    several settings is conducted in this file

- files concerned with Docker and git
- requirements
    
    all the libraries to execute this project

## Project Install

- Environment
    
    Balena OS, python 3.6

- In Jetson Nano
    
    Please download the image of Balena application where this project is pushed and build it in your Jetson Nano board.  

- In local environment
    
    * System: Ubuntu 18.04
    * Install python 3.6
    * In the directory of this project, run following command in terminal:
    
    ```
        pip3 install -r requirements.txt
    ``` 
    
    * Download the models to detect person from https://drive.google.com/file/d/10AXoNg4y6U1fwvvlhkjxHrXf3b4GXdtb/view?usp=sharing and 
    copy them to src/person. Also download the models to estimate age and gender from and copy them to src/age_gender.

## Project Execution

- In Jetson Nano
    
    Once building image of the corresponding application is finished, this project is automatically run in Jetson Nano.

- In local env
    
    * Please go ahead the directory of this project and run the following command.
    
    ```
        python3 src/app.py
    ```
