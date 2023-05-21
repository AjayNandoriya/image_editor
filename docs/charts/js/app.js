const Plot = createPlotlyComponent(Plotly);

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {data1: true,data2: true, df:null, layout:{}};
      }
    draw_time_chart() {

        if(!this.state.df){
            return [];
        }

        const df = this.state.df.sortValues("timestamp");
        const sub_df1 = df.loc({rows:df.id.eq("abc1")});
        const sub_df2 = df.loc({rows:df.id.eq("abc2")});

        let trace1 = {
            x: sub_df1.timestamp.values,
            y: sub_df1.index1.values,
            type: "scatter",
            mode: "lines+markers",
            name: "data1",
            text: sub_df1.category3.values,
            marker:{
                color:"red"
            }

        }
        let trace2 = {
            x: sub_df1.timestamp.values,
            y: sub_df2.index1.values,
            type: "scatter",
            mode: "lines+markers",
            name: "data2",
            text: sub_df2.category3.values,
            marker:{
                color:"blue"
            }
        }

        let data = [];
        if(this.state.data1){
            data.push(trace1);
        }
        if(this.state.data2){
            data.push(trace2);
        }
        
        let layout = {
            title: 'Line and Scatter Plot',
            xaxis: {
                title: "time -->",
                showgrid: true,

            },
            yaxis: {
                title: "value",

            }
        };

        return data;
    }

    async loadData(){
        const df = await dfd.readCSV("./data/sample.csv");
        this.setState({
            df:df
        });
        
    }
    render() {
        return (
            <div>
                <button onClick={() => this.loadData()}>Load</button>
                <input type="checkbox" onChange={(e)=>this.setState({data1: e.target.checked})} name="data1" defaultChecked={this.state.data1} ></input>
                <input type="checkbox" onChange={(e)=>this.setState({data2: e.target.checked})} name="data2" defaultChecked={this.state.data2} ></input>
                <Plot data={this.draw_time_chart()} layout={this.state.layout} scrollZoom="true"></Plot>
            </div>
        );
    }
}
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
