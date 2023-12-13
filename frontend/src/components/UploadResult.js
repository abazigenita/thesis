import React from "react";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import Generate from "./Generate";

function UploadResult({similarPapers, onClose, originalAbstract}) {
    const isArxivId = (id) => {
        return /^\d+\.\d+$/.test(id);
    };

    const renderSimilarPapers = () => {
        return similarPapers.map((paper, index) => (
            <div key={index} className="upload-results__papers">
                <h3>{paper.title}</h3>
                <p className="upload-results__papers__authors"><span>Authors</span>: {paper.authors}</p>
                <span className="upload-results__papers__date">{paper.date}</span>
                <p>{paper.abstract}</p>
                {isArxivId(paper.id) &&
                    <a href={`https://arxiv.org/abs/0${paper.id}`}
                       target="_blank"
                       rel="noreferrer"
                       className="upload-results__papers__link"
                    >
                        <FontAwesomeIcon icon="fa-solid fa-arrow-up-right-from-square"/> Paper Details
                    </a>
                }
                <hr/>
            </div>
        ));
    };

    return (
        <>
            <div className="upload-results">
                <div className="upload-results__all">
                    <h2>Similar Papers to Your Work</h2>
                    <hr/>
                    <br/>
                    {similarPapers.length > 0 && renderSimilarPapers()}
                    <Generate originalAbstract={originalAbstract} similarPapers={similarPapers}/>
                </div>
                <button className="upload-results__close" onClick={onClose}>
                    X
                </button>
            </div>
        </>
    );
}

export default UploadResult;
