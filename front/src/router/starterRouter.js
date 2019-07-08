import Vue from 'vue';
import Router from 'vue-router';
import DashboardLayout from '../layout/starter/SampleLayout.vue';
import Starter from '../layout/starter/SamplePage.vue';

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
          components: { default: Starter }
        }
      ]
    }
  ]
});
