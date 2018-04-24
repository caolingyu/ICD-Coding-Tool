import Vue from 'vue'
import App from './App.vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import axios from "axios"
import router from './router'


Vue.use(ElementUI);
Vue.prototype.$axios = axios;

new Vue({
  el: '#app',
  router: router,
  render: h => h(App)
})
