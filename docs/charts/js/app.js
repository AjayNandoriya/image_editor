const Plot = createPlotlyComponent(Plotly);

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {data1: true,data2: true, layout:{}};
      }
    draw_time_chart() {

        let trace1 = {
            x: ['2013-10-04 22:23:00', '2013-11-04 22:23:00', '2013-12-04 22:23:00'],
            y: [1, 4, 3],
            type: "scatter",
            mode: "lines+markers",
            name: "data1",
            text: ["a", "b", "c"],
            marker:{
                color:"red"
            }

        }
        let trace2 = {
            x: ['2013-10-04 22:23:00', '2013-11-04 22:23:00', '2023-12-04 22:23:00'],
            y: [1, -1, 3],
            type: "scatter",
            mode: "lines+markers",
            name: "data2",
            text: ["x", "y", "z"],
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

    
    render() {
        return (
            <div>
                <button onClick={() => this.draw_time_chart()}>Draw</button>
                <input type="checkbox" onChange={(e)=>this.setState({data1: e.target.checked})} name="data1" defaultChecked={this.state.data1} ></input>
                <input type="checkbox" onChange={(e)=>this.setState({data2: e.target.checked})} name="data2" defaultChecked={this.state.data2} ></input>
                <Plot data={this.draw_time_chart()} layout={this.state.layout} scrollZoom="true"></Plot>
            </div>
        );
    }
}
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
