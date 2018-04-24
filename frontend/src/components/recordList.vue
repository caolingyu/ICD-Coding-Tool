<template>
  <div id="recordList">
      <!-- <h1>输入记录</h1>
      <ul v-for="(item,index) in getInputList"
          :key=index>
        <el-button type="text" @click="open(index)">{{ item }}</el-button>
      </ul> -->
      <el-menu :default-openeds="['1','2']" background-color="#D3DCE6" style="text-align: left; margin-left: 20px">
        <el-submenu index="1" style="display: inline">
            <template slot="title"><i class="el-icon-edit"></i>输入记录</template>
            <!-- <el-menu-item-group> -->
            <el-menu-item :index="'1'+index" v-for="(item,index) in getInputList" :key=index @click="open(index)">{{ item }}</el-menu-item>
            <!-- <el-menu-item index="1-2">选项2</el-menu-item> -->
            <!-- </el-menu-item-group> -->
        </el-submenu>
        <el-submenu index="2" style="display: inline">
            <template slot="title"><i class="el-icon-document"></i>上传记录</template>
            <el-menu-item :index="'2'+index" v-for="(item,index) in getUploadList" :key=index @click="open2(index)">{{ item }}</el-menu-item>
        </el-submenu>
      </el-menu>
  </div>
</template>

<script>
    import {mapGetters, mapActions} from 'vuex';
    export default {
        computed: {
            getInputList (){
                return this.$store.state.inputList;
            },
            getUploadList (){
                // console.log(this.$store.state.uploadList)
                return this.$store.state.uploadList;
            }
        },

        methods: {
            open (index){
                this.$store.dispatch("changeIfUpload", false)
                this.$store.dispatch("changeCurrentIndex", index)
            },
            open2 (index){
                this.$store.dispatch("changeIfUpload", true)                
                this.$store.dispatch("changeCurrentIndex2", index)
            }
        }
    }
</script>

<style>
    .el-submenu .el-menu-item {
      min-width: 0;
      margin:0 auto
    }
</style>