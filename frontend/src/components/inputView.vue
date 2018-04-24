<template>
    <div class='display' v-if="getSwitch2">
        <!-- {{ gettersInputData[this.$store.state.currentIndex] }} -->
        <span v-for="(item, index) in getRow" :key=index :style="{'background-color': item.nerColor}">{{item.text}}</span>
    </div>
    <div class='display' v-else>
        <span v-for="(item, index) in getRow" :key=index>{{item.text}}</span>
    </div>
</template>

<script>
    import {mapGetters, mapActions} from 'vuex';
    export default {
        data(){
            return{
            cur: []
            }
        },
        methods: {
            setNerColor(text) {
                switch(text)
                {
                case "症状和体征(+)":
                return '#F56C6C'
                break;
                case "症状和体征(-)":
                return '#EBB9B9'
                break;
                case "检查和检验":
                return '#E6A23C'
                break;
                case "身体部位":
                return '#67C23A'
                case "疾病和诊断":
                return '#409EFF'
                break;
                case "治疗":
                return '#909399'
                break;
                default:
                return '#E9EEF3'
                }
            }
        },

        computed: {
            // ...mapGetters(['gettersInputData'])
            getData() {
                var _ifUpload = this.$store.state.ifUpload
                if (_ifUpload == true){
                    return this.$store.state.tableData2[this.$store.state.currentIndex2];
                }
                else{
                    return this.$store.state.tableData[this.$store.state.currentIndex];
                }
                
            },

            getRow() {
                this.cur = []
                var _currentRow = this.$store.state.currentRow
                var _currentIndex = null
                var _tableData = null
                if (this.$store.state.ifUpload == true){
                    _currentIndex = this.$store.state.currentIndex2
                    _tableData = this.$store.state.tableData2    
                }
                else{
                    _currentIndex = this.$store.state.currentIndex
                    _tableData = this.$store.state.tableData
                }
                if (_currentRow != null){
                    // console.log(_currentIndex)
                    for (var i=0;i<_currentRow.textRaw.length;i++){
                        // console.log(this.$store.state.currentRow.alpha[i])
                        var _c = 'rgba(28, 129, 224, ' + _currentRow.alpha[i].toFixed(2) + ')'
                        var _NerC = this.setNerColor(_currentRow.ner[i])
                        this.cur.push({text: _currentRow.textRaw[i], color: _c, nerColor: _NerC})
                    }
                    // return this.$store.state.currentRow;
                }
                else {
                    // console.log(_tableData)
                    if (_tableData.length != _currentIndex && this.$store.state.ifDone==true) {
                        for (var i=0;i<_tableData[_currentIndex][0].textRaw.length;i++){
                            var _NerC = this.setNerColor(_tableData[_currentIndex][0].ner[i])
                            // console.log(_NerC)
                            this.cur.push({text: _tableData[_currentIndex][0].textRaw[i], color: '#E9EEF3', nerColor: _NerC})
                        }
                        // return this.$store.state.tableData[this.$store.state.currentIndex][0]
                    }
                }
                // console.log(this.cur)
                return this.cur
                // return this.$store.state.currentRow
            },

            getSwitch1() {
                return this.$store.state.switch1;
            },

            getSwitch2() {
                return this.$store.state.switch2;
            }

            // getCur() {
            //     return this.cur
            // }
        }
    }
    
</script>

<style>
    .display {
        border: 5px solid #D3DCE6; 
        width: 30rem; 
        height: 20rem; 
        overflow: scroll
    }
</style>