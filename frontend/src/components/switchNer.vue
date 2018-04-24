<template>
    <el-switch
    v-model="value"
    active-text="显示实体"
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
            this.$store.dispatch("changeSwitch2", this.value)
        }
    }

  };
</script>