# Bert_Deployment
## Using a trained model and an input.csv file to predict in a docker container 

<details open>
<summary> Prepare your local folder shown below </summary>
  
Bert_deployment
  - input.csv
  - Dockerfile
  - model.pt (have a look from here ......)
  - config.py
  - main.py
  - predict.py
  - requirements.txt
  - to_load.py
  - gitignore( with model.pt inside)


<details open>
<summary>Build and run docker Image</summary>
  
  - open your Docker desktop 
  - create a new empty folder (e.g. classi) and put in it:
       .  the Dockerfile
       .  the model.pt (they will be copied to docker container using the code in the dockerfile)
  - build your image

   ```bash
   docker build -t your image name:your tag . 
   ```
  - run the image container 
  ```bash
   docker run your image name:your tag
  ```
  <details open>
  <summary>What happens when you run the container </summary>
    
  - The files in the github repository will be pushed automatcillay to the docker container
  - the prediction will be run inside the container, printed in the cmd (bash in my case)
  - the model.pt will be copied as well from locally and a new csv file will be created in the container: it's the output.csv that includes the prediction as requested by predict.py
  - if you want to see the result directly in the container you can replace the previous line by the next one  (300, 200, etc : the the duration you want the container keeps alive
    
   ```bash
   docker run your image name:your tag  sleep 3000
   ```
  
 Use this code to print information about the containers
  
   ```bash
   docker ps -a
  ```
  <details open>
  <summary>The container CLI </summary>
    
  In order to see the files indide the running container :
  - take the container id from the output of the previous code and look for it in
  the docker app. 
  - click on the running container CLI and see the files by typing ls

 
<details open>
<summary>Copy your output in your local host</summary>
  
 We need to copy outpu.csv from the repository Bert_deployment in the container to our classi folder which has for example the path /c/Users/manar/Desktop/classi
  get the running container id or name (using the line docker ps -a)
  
  ```bash
  docker cp container id:/home/Bert_deployment/ouptput.csv /c/Users/manar/Desktop/classi
  ```

Find your output.csv file in your local classi folder !
