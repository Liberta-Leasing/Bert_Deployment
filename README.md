# Bert_Deployment
# Using Bert model Docker
## Using your own trained model and your input.csv file  
<details open>
<summary>Install and Prepare</summary>
  
```bash
git clone https://github.com/Abzurde/yolov5.git
```
<details open>
<summary> Prepare your local folder shown below </summary>
* Bert_deployment
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
  --> the prediction will be run inside the container
  <details open>
  <summary>explore the container </summary>
  * The files in the github repository will be pushed automatcillay to docker container, the model.pt will be copied as well from locally and new file will be created in the container: it's the output.csv that includes the prediction as requested by predict.py.
  
 * Use this code to print information about the containers
  
   ```bash
   docker ps -a
  ```
  * In order to see the files indide the running container :
  - take the container id from the output of the previous code and look for it in
  the docker app. 
  - click on the running container CLI and see the files by typing ls

 
<details open>
<summary>Copy your output in your local host</summary>
  
We need to copy outpu.csv from the repository Bert_deployment in the container to our classi folder which has the path /c/Users/manar/Desktop/classi
  get the running container id or name (using the line docker ps -a)
  
```bash
docker cp containerid:/home/Bert_deployment/ouptput.csv /c/Users/manar/Desktop/classi
```

your output.csv file is now in your classi folder
