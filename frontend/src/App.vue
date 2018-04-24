<template>
  <div id="app">
    <!-- <basic></basic> -->
    <router-view></router-view>
  </div>
</template>

<script>
// import basic from './components/basic.vue'
import store from './store/index' //后面讲vuex 配置会提到

export default {
  name: 'app',
  data () {
    return {
    }
  },
  store
  // components: { basic }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 0px;
  height: 100%;
}

h1, h2 {
  font-weight: normal;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  display: inline-block;
  margin: 0 10px;
}

a {
  color: #42b983;
}
</style>
