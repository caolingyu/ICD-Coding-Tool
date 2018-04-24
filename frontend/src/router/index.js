import Vue from 'vue';
import Router from 'vue-router';
import VueResource from 'vue-resource';
import index from '../components/index.vue';

Vue.use(Router);
Vue.use(VueResource);

export default new Router({
    mode: 'history',
    base: __dirname,
    routes: [
        {
        path: '/',
        name: 'index',
        component: index
        }
    ]
})