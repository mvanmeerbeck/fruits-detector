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