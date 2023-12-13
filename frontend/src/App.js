import React, {useState, useEffect} from "react";
import axios from "axios";
import Search from "./components/Search";
import Filter from "./components/Filter";
import Upload from "./components/Upload";
import Map from "./components/Map";
import Loader from "./components/Loader";
import UploadResult from "./components/UploadResult";
import Details from "./components/Details";
import "./style/main.css";

function App() {
    const [plotData, setPlotData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showFilter, setShowFilter] = useState(false);
    const [similarPapers, setSimilarPapers] = useState([]);
    const [showUploadResults, setShowUploadResults] = useState(false);
    const [onlyLicensed, setOnlyLicensed] = useState(false);
    const [originalAbstract, setOriginalAbstract] = useState("");
    const [selectedDataPoint, setSelectedDataPoint] = useState(null);
    const [isDetailsVisible, setIsDetailsVisible] = useState(false);

    const toggleFilter = () => {
        setShowFilter(!showFilter);
    };

    useEffect(() => {
        axios
            .get("http://localhost:5002/get-data")
            .then((response) => {
                setPlotData(response.data);
                setLoading(false);
            })
            .catch((error) => {
                console.error("Error fetching plot data:", error);
                setLoading(false);
            });
    }, []);

    const handleSearchResults = (results) => {
        setPlotData(results);
    };

    const handleFilterResults = (filteredResults) => {
        setPlotData(filteredResults);
    };

    const handleCloseUploadResults = () => {
        setShowUploadResults(false);
    };

    const handleNewData = (newData) => {
        setPlotData([...plotData, newData]);
        setSimilarPapers(newData.similarPapers || []);
        setOriginalAbstract(newData.abstract);
        setShowUploadResults(true);
    };

    const handleDataPointClick = (dataPoint) => {
        setSelectedDataPoint(dataPoint);
        setIsDetailsVisible(true);
    };

    const handleCloseDetails = () => {
        setIsDetailsVisible(false);
    };

    return (
        <>
            <Search
                onSearch={handleSearchResults}
                onToggleFilter={toggleFilter}
                setLoading={setLoading}
                showFilter={showFilter}
            />
            {showFilter && <Filter
                onlyLicensed={onlyLicensed}
                setOnlyLicensed={setOnlyLicensed}
                onFilter={handleFilterResults}
                setLoading={setLoading}
            />}
            <Upload onNewData={handleNewData} setLoading={setLoading}/>
            {showUploadResults && (
                <UploadResult
                    similarPapers={similarPapers}
                    onClose={handleCloseUploadResults}
                    originalAbstract={originalAbstract}
                />
            )}
            {loading ? (
                <Loader isLoading={loading}/>
            ) : (
                plotData.length > 0 && (
                    <div className="plot-container">
                        <Map data={plotData} onDataPointClick={handleDataPointClick}/>
                    </div>
                )
            )}

            {isDetailsVisible && selectedDataPoint && (
                <Details selectedData={selectedDataPoint} onClose={handleCloseDetails}/>
            )}
        </>
    );
}

export default App;
