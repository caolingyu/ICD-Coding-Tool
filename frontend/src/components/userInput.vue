<template>
  <div style="display: inline">
    <el-button type="primary" plain size="small" @click="open" sytle="margin: 0 auto">输入数据</el-button>
  </div>
</template>

<script>
  import axios from 'axios'
  import {mapActions} from "vuex";
  import { Loading } from 'element-ui';


  export default {
    data () {
      return{
        loading: null
      }
    },

    methods: {
      open() {
        this.$prompt('请输入文本数据', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          inputType: "textarea",
          inputPattern: /\S+/,
          inputErrorMessage: '格式不正确'
        }).then(({ value }) => {
          this.$message({
            type: 'success',
            message: '输入成功'
          });
          let options = {
            lock: true,
            text: 'Loading',
            spinner: 'el-icon-loading',
            background: 'rgba(0, 0, 0, 0.7)'
          }
          this.loading = Loading.service(options);
          
          this.$store.dispatch("changeInputData", value);
          var _len = this.$store.state.inputList.length + 1
          this.$store.dispatch("changeInputList", "INPUT"+_len)
          this.$store.dispatch("changeCurrentIndex", _len-1)
          this.sendData(value);
        }).catch(() => {
          this.$message({
            type: 'info',
            message: '取消输入'
          });       
        });
      },

      // sendData(value){
      //     this.$store.dispatch("changeTableLoading", true)
      //     const path = `http://localhost:5000/api/test`
      //     axios.post(path, {
      //     data: value
      //     })
      //     .then(response => {
      //         // this.$store.dispatch("resetTableData")
      //         this.$store.dispatch("changeTableData", response)
      //         this.$store.dispatch("changeTableLoading", false)
      //         this.$store.dispatch("changeIfUpload", false)

      //     }).catch(error => {
      //         console.log(error)
      //     });
      // }
      
      sendData(value){
          this.$store.dispatch("changeTableLoading", true)
          const path = `http://localhost:5000/api/test`
          const path2 = `http://localhost:5001/ner_api`
          axios.all([
            axios.post(path, {data: value}),
            axios.post(path2, {content: value})
          ]).then(axios.spread((tableResp, nerResp) => {
              // Handling table data
              this.$store.dispatch("changeTableData", tableResp)
              this.$store.dispatch("changeTableLoading", false)
              this.$store.dispatch("changeIfUpload", false)
              // Handling input view data
              nerResp.state = this.$store.state
              this.$store.dispatch("addNER", nerResp)
          
              this.$store.dispatch("changeIfDone", true)
                          
              this.loading.close();
              this.$message({
                  message: '编码完成',
                  type: 'success'
              });

              })).catch(error => {
                  console.log(error)
              });
          }

    }
  }
</script>