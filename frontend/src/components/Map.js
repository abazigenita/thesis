import React from "react";
import Plot from "react-plotly.js";

const Map = React.memo(({data, onDataPointClick}) => {
    const handleClick = (event) => {
        if (event.points && event.points.length > 0) {
            const pointIndex = event.points[0].pointIndex;
            const selectedPoint = data[pointIndex];
            onDataPointClick(selectedPoint); // Call the passed function
        }
    };

    const getMarkerSize = (item) => {
        if (item.isNew) {
            return 15;
        }
        return item.normalized_distance ? item.normalized_distance * 10 + 5 : 10;
    };

    const getMarkerColor = (item) => {
        if (item.isNew) {
            return 'black';
        }
        return item.cluster !== undefined ? item.cluster.toString() : 'defaultColor';
    };

    const plotData = [
        {
            x: data.map((item) => item.dimension_X),
            y: data.map((item) => item.dimension_Y),
            type: "scatter",
            mode: "markers",
            marker: {
                size: data.map(getMarkerSize),
                color: data.map(getMarkerColor),
            },
            text: data.map((item) => `Title: ${item.title}<br>Category: ${item.categories}`),
        },
    ];

    const layout = {
        title: "UMAP 2D Visualization",
        xaxis: {
            title: "",
            showgrid: false,
            showline: false,
            zeroline: false
        },
        yaxis: {
            title: "",
            showgrid: false,
            showline: false,
            zeroline: false
        },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        showlegend: false,
        autosize: true,
        margin: {l: 0, r: 0, b: 0, t: 0},
    };

    return (
        <>
            <Plot
                data={plotData}
                layout={layout}
                onClick={handleClick}
            />
        </>
    )
});

export default Map;
