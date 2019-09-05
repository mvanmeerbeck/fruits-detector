# fruits-detector 

Real time, scalable fruits detector

## Requirements 
- RTMP Camera
- Docker
- Docker-compose

## Install
### RTMP Camera
Install the android application RTMP Camera

https://play.google.com/store/apps/details?id=com.miv.rtmpcamera&hl=fr

## Start

Start the containers

```docker-compose up -d```

Find your IP address

```sudo ifconfig```

Open the RTMP Camera on your smartphone and change the RTMP server in Options > Publish address to

```rtmp://your-ip/live```

Check the result with VLC via "Open Network Stream" with the address

```rtmp://your-ip/live```

## log
### 29/06
- réflexion, recherche sur l'architecture
- tests de nginx-rtmp

### 06/07
- installation env python (docker, requirements)
- premier tests chargement des données, training
- init du script de prédiction "real time"

### 05/09
- test du model inceptionv3