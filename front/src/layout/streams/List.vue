<template>
  <div>
    <div v-for="streamsRow in streamsRows" class="row">
      <div v-for="stream in streamsRow" class="col-lg-4">
        <div class="card card-chart">
          <div class="card-header">
            <h5 class="card-category">Stream</h5>
            <h3 class="card-title"><i class="tim-icons icon-video-66 text-primary"></i>
              <router-link :to="{ name: 'stream', params: { name: stream.name }}">{{ stream.name }}</router-link>
            </h3>
          </div>
          <div class="card-body">
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
  import axios from 'axios';
  import _ from 'lodash';

  export default {
    name: 'streams-list',
    data() {
      return {
        streams: null
      };
    },
    mounted() {
      axios
        .get('http://fruits-detector.com:5000/streams')
        .then(response => {
          this.streams = response.data
        });
    },
    computed: {
      streamsRows() {
        return _.chunk(this.streams, 3)
      }
    }
  };
</script>
<style></style>
