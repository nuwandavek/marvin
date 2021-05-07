# ML
## Running the App
```
#Without Docker
python ml_server.py

#With Docker
bash run_ml_docker.sh
```

## Build the docker images
```
bash ../build_docker_images.sh
```
---

## Bucket Details
- Shakespeare : low = [0,0.1]; mid = [0.1,0.9]; high = [0.9,1]
- Formality : low = [0,0.2]; mid = [0.4,0.7]; high = [0.9,1]; skip the rest
- Emo : low = [0,0.25]; mid = [0.4,0.7]; high = [0.9,1]; skip the rest