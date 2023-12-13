import React, {useState} from "react";
import axios from "axios";
import Loader from "./Loader";
import Abstract from "./Abstract";

function Generate({originalAbstract, similarPapers}) {
    const [generatedAbstract, setGeneratedAbstract] = React.useState("");
    const [loading, setLoading] = useState(false);
    const [showAbstract, setShowAbstract] = useState(false);

    const handleGenerateAbstract = async () => {
        setLoading(true);

        try {
            const similarAbstracts = similarPapers.map(paper => paper.abstract);

            const response = await axios.post('http://localhost:5002/generate-abstract', {
                original_abstract: originalAbstract,
                similar_abstracts: similarAbstracts
            });

            setGeneratedAbstract(response.data.generated_abstract);
            setShowAbstract(true);
        } catch (err) {
            console.log("Error generating abstract: " + err.message);
            setGeneratedAbstract(""); // Reset generated abstract in case of error
        } finally {
            setLoading(false);
        }
    };

    const handleCloseAbstract = () => {
        setShowAbstract(false);
    };

    return (
        <>
            <div className="generate__wrapper">
                {loading ? (
                    <Loader isLoading={loading}/>
                ) : (
                    <>
                        <button onClick={handleGenerateAbstract} className="generate__button">
                            Generate New Abstract
                        </button>
                    </>
                )}
            </div>
            {showAbstract && (
                <Abstract
                    generatedAbstract={generatedAbstract}
                    onClose={handleCloseAbstract}
                />
            )}
        </>
    );
}

export default Generate;
