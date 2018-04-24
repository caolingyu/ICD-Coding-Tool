<template>
  <div id="main">
    <el-container style="border: 1px solid #eee">
      <el-header style="text-align: left; font-size: 20px">An ICD Coding Tool</el-header>
      <el-container>
        <el-aside width="200px"></el-aside>
        <el-main>
          <el-row :gutter="100" type="flex" justify="space-around">
            <el-col :span="200">
              <el-form ref="form" :model="form" label-width="80px">
                <el-form-item>
                  <el-input 
                    type="textarea" 
                    rows="6"
                    resize="none"
                    style="width: 20rem"
                    v-model="form.context">
                  </el-input>
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" @click="onSubmit" :disabled="form.context===''">提交</el-button>
                  <el-button @click="reset" :disabled="form.context===''">重置</el-button>
                </el-form-item>
              </el-form>
            </el-col>
            <el-col :span="500">
                <el-table
                ref="singleTable"
                :data="tableData"
                highlight-current-row
                @current-change="handleCurrentChange"
                style="width: 100%"
                :row-class-name="tableRowClassName"
                v-loading="loading">
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
            </el-col>
          </el-row>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>
 
<script>
import axios from 'axios'

export default {
  data () {
    return {
        form: {
          context: ''
        },
        results: '',
        prob: '',
        desc: '',
        tableData: [],
        currentRow: null,
        loading: false,
        dialogVisible: false,
        codeSelected: ''
    }
  },
  methods: {
      onSubmit() {
        this.sendData()
        this.loading = true
      },

      sendData(){
        this.new_results
        const path = `http://localhost:5000/api/test`
        axios.post(path, {
          data: this.form.context
        })
        .then(response => {
            this.tableData = []
            this.results = response.data.results
            this.prob = response.data.prob
            this.desc = response.data.desc
            for (var i=0;i<this.results.length;i++){
              this.tableData.push({code: this.results[i],
                                   prob: this.prob[i].toFixed(2),
                                   desc: this.desc[i]})
            }
            this.loading = false
        }).catch(error => {
            console.log(error)
        });
      },

      reset(){
        this.results = ''
        this.form.context = ''
      },

      setCurrent(row) {
        this.$refs.singleTable.setCurrentRow(row);
        this.currentRow = null
      },

      handleCurrentChange(val) {
        this.currentRow = val;
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