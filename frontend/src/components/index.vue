<template>
  <div id="index" style="height: 100%">
    <el-container style="border: 1px solid #eee; height: 100%">
        <el-header style="text-align: left; font-size: 20px">
          <el-row :gutter="0" type="flex" justify="space-between">
            <el-col :span="500">
              <h1 style="margin:0 auto">An ICD Coding Tool</h1>
            </el-col>
            <el-col :span="500">
              <!-- <switchAttn></switchAttn> -->
              <switchNer></switchNer>
            </el-col>
          </el-row>
        </el-header>
      <el-container>
        <el-aside width="200px">
          <div style="margin: 5px">
            <userInput></userInput>
            <userUpload></userUpload>
          </div>
            <hr/>
            <recordList></recordList>
            <hr/>
        </el-aside>
        <el-main>
          <el-row :gutter="0" type="flex" justify="space-around">
            <el-col :span="500">
                <inputView></inputView>
                <nerTag></nerTag>
            </el-col>
            <el-col :span="500">
                <resultTable></resultTable>
            </el-col>
          </el-row>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import axios from 'axios'
import userUpload from './userUpload.vue'
import inputView from './inputView.vue'
import resultTable from './resultTable.vue'
import userInput from './userInput.vue'
import recordList from './recordList.vue'
import switchAttn from './switchAttn.vue'
import nerTag from './nerTag.vue'
import switchNer from './switchNer.vue'

export default {
  data () {
    return {
    }
  },

  components: { 
      userUpload,
      inputView,
      resultTable,
      userInput,
      recordList,
      switchAttn,
      nerTag,
      switchNer
  }
}
</script>
 
<style>
  .el-header, .el-footer {
    background-color: #B3C0D1;
    color: #333;
    text-align: center;
    line-height: 60px;
  }

  .el-aside {
    background-color: #D3DCE6;
    color: #333;
    text-align: center;
    /* line-height: 200px; */
  }

  .el-main {
    background-color: #E9EEF3;
    color: #333;
    text-align: center;
    /* line-height: 200px; */
  }

  body > .el-container {
    margin-bottom: 40px;
  }

  .el-container:nth-child(5) .el-aside,
  .el-container:nth-child(6) .el-aside {
    line-height: 260px;
  }

  .el-container:nth-child(7) .el-aside {
    line-height: 320px;
  }
</style>