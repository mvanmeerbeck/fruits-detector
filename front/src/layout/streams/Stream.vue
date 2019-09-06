<template>
  <div>
    <div class="row">
      <div class="col-lg-8">
        <div class="card card-chart">
          <div v-if="stream" class="card-header">
            <h5 class="card-category">Stream</h5>
            <h3 class="card-title"><i class="tim-icons icon-video-66 text-primary"></i>{{ stream.name }}</h3>
          </div>
          <div class="card-body">
            <video ref="video" id="video" width="100%" controls autoplay></video>
          </div>
        </div>
      </div>
      <div v-if="stream" class="col-lg-4">
        <div class="card card-chart">
          <div class="card-header">
            <h5 class="card-category">Prédiction</h5>
            <h3 class="card-title"><i class="text-primary"></i>{{ stream.prediction }}</h3>
          </div>
          <div class="card-body">
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
  import Hls from 'hls.js';
  import axios from 'axios';
  import _ from 'lodash';

  export default {
    name: 'stream',
    data() {
      return {
        stream: null,
        polling: null
      };
    },
    mounted() {
      const video = this.$refs.video;
      const hls = new Hls();

      hls.loadSource('http://fruits-detector.com:8080/live/' + this.$route.params.name + '.m3u8');
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED,function() {
        video.play();
      });
    },
    methods: {
      pollData () {
        this.polling = setInterval(() => {
          axios
            .get('http://fruits-detector.com:5000/streams/' + this.$route.params.name)
            .then(response => {
              this.stream = response.data;
            });
        }, 1000);
      }
    },
    beforeDestroy () {
      clearInterval(this.polling);
      video.stop();
    },
    created () {
      this.pollData()
    }
  };
</script>
<style></style>
