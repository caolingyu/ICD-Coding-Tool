<template>
  <div id="resultTable">
    <el-table
    ref="singleTable"
    :data="tableData"
    highlight-current-row
    @current-change="handleCurrentChange"
    style="width: 100%"
    :row-class-name="tableRowClassName">
    <el-table-column
        type="index"
        width="50">
    </el-table-column>
    <el-table-column
        property="code"
        label="候选编码"
        width="120">
    </el-table-column>
    <el-table-column
        property="desc"
        label="编码描述"
        width="200">
    </el-table-column>
    <el-table-column
        property="prob"
        label="概率"
        width="120">
    </el-table-column>
    </el-table>
    <div style="margin-top: 20px">
    <el-button :plain="true" @click="confirmSelection" :disabled="currentRow===null">确认</el-button>
    <el-button @click="setCurrent()" :disabled="currentRow===null">取消选择</el-button>
    </div>
    <el-dialog
    title="编码确认"
    :visible.sync="dialogVisible"
    width="30%"
    :show-close="false"
    :center="true">
    <span style="width:100%;text-align:center;display:block;">您选择了{{ codeSelected }}，请确认。</span>
    <span slot="footer" class="dialog-footer">
        <el-button type="primary" @click="confirmDialog">确认</el-button>
        <el-button @click="dialogVisible=false">取消</el-button>
    </span>
    </el-dialog>
  </div>
</template>
 
<script>
import axios from 'axios'
import {mapGetters, mapActions} from 'vuex';

export default {
  data () {
    return {
        form: {
          context: ''
        },
        results: '',
        prob: '',
        desc: '',
        // tableData: [],
        currentRow: null,
        // loading: false,
        dialogVisible: false,
        codeSelected: ''
    }
  },

  methods: {
      reset(){
        this.results = ''
        this.form.context = ''
      },

      setCurrent(row) {
        this.$refs.singleTable.setCurrentRow(row);
        this.currentRow = null
        this.$store.dispatch("changeCurrentRow", null);
      },

      handleCurrentChange(val) {
        this.currentRow = val;
        this.$store.dispatch("changeCurrentRow", val);
      },

      confirmSelection(){
        this.dialogVisible = true
        this.codeSelected = this.currentRow.code
      },

      confirmDialog(){
        this.dialogVisible = false
        this.$message({
          message: '编码已成功提交',
          type: 'success'
        });
      },

      tableRowClassName({row, rowIndex}) {
        if (row.prob <= 1 && row.prob >= 0.8) {
          return 'level1';
        } else if (row.prob < 0.8 && row.prob >= 0.6) {
          return 'level2';
        } else if (row.prob < 0.6 && row.prob >= 0.4) {
          return 'level3';
        } else if (row.prob < 0.4 && row.prob >= 0.2) {
          return 'level4';
        } else if (row.prob < 0.2 && row.prob >= 0) {
          return 'level5';
        }
        return '';
      }
    },

    computed:{
        tableData() {
            // console.log(this.$store.state.tableData)
            var _ifUpload = this.$store.state.ifUpload
            if (_ifUpload == true){
                return this.$store.state.tableData2[this.$store.state.currentIndex2];
            }
            else{
                return this.$store.state.tableData[this.$store.state.currentIndex];
            }
            // return this.$store.state.tableData[this.$store.state.currentIndex];
        },

        loading() {
            return this.$store.state.tableLoading;
        }
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
  
  .el-table th{
    text-align: center;
  }

  .el-table .level1 {
    background: rgb(228, 167, 161);
  }

  .el-table .level2 {
    background: #f0e4a3;
  }

  .el-table .level3 {
    background: #cdeea8;
  }

  .el-table .level4 {
    background: #b1eeee;
  }

  .el-table .level5 {
    background: #d0d4ec;
  }
</style>