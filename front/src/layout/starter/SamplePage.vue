<template>
  <div>
    <div v-for="streamsRow in streamsRows" class="row">
      <div v-for="stream in streamsRow" class="col-lg-4">
        <div class="card card-chart">
          <div class="card-header">
            <h5 class="card-category">Stream</h5>
            <h3 class="card-title"><i class="tim-icons icon-video-66 text-primary"></i>{{ stream.name }}</h3>
          </div>
          <div class="card-body">
            <!-- Or if you want a more recent canary version -->
            <!-- <script src="https://cdn.jsdelivr.net/npm/hls.js@canary"></script> -->
            <!--<video ref="video" id="video" width="100%" controls></video>-->
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
  //  import Hls from 'hls.js';
  import axios from 'axios';
  import _ from 'lodash';

  export default {
    name: 'starter-page',
    data() {
      return {
        streams: null
      };
    },
    mounted() {
      axios
        .get('http://localhost:5000/streams')
        .then(response => {
          this.streams = response.data
        });

//      var video = this.$refs.video;
//      var hls = new Hls();
//      hls.loadSource('http://localhost:8080/live/test.m3u8');
//      hls.attachMedia(video);
//      hls.on(Hls.Events.MANIFEST_PARSED,function() {
//        video.play();
//      });
    },
    computed: {
      streamsRows() {
        return _.chunk(this.streams, 3)
      }
    }
  };
</script>
<style></style>
