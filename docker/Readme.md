### Build the docker image from within folder
```bash
docker build --rm --platform linux/amd64 -t ml:1.0 .
```

### This changes the name of the image
```bash
docker tag d9ef125cbd79 evafast1/ml:1.0
```

### this does the login for uploading to dockerhub
```bash
docker login -u evafast1 -p <password>
```

### this pushes to the repo
```bash
docker push evafast1/ml:1.0
```
## **START HERE** if you are not rebuilding container:

1) install docker desktop from website (should be free)

2) Launch the docker image
at the -v flag change the first part to you directory on your computer that contains the data
```bash
docker run \
--rm \
-d \
--name ml \
-p 8881:8888 \
-e JUPYTER_ENABLE_LAB=YES \
-v /Users/eva/Library/CloudStorage/OneDrive-Personal/Documents:/home/jovyan/work \ 
evafast1/ml:1.0
```
3) check if the docker container has launched
``` bash
docker ps
```

4) access the tocken for launching the container
``` bash
docker logs ml
```

5) copy the localhost IP adress (should be something starting with 127.0.0.0:8881 etc into your browser, making sure that you change the numbers after the column from 8888 to 8881
