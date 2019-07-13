import Vue from 'vue';
import Router from 'vue-router';
import DashboardLayout from '../layout/streams/SampleLayout.vue';
import List from '../layout/streams/List.vue';
import Stream from '../layout/streams/Stream.vue';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'streams',
      redirect: '/streams',
      component: DashboardLayout,
      children: [
        {
          path: 'streams',
          name: 'streams',
          components: { default: List }
        },
        {
          path: 'streams/:name',
          name: 'stream',
          components: { default: Stream }
        },
      ]
    }
  ]
});
