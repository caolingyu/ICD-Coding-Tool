<template>
    <el-switch
    v-model="value"
    active-text="显示权重"
    :change="changeSwitch()">
    </el-switch>
</template>

<script>
  export default {
    data() {
      return {
          value: 'false'
      }
    },

    methods: {
        changeSwitch(){
            this.$store.dispatch("changeSwitch1", this.value)
        }
    }

  };
</script>