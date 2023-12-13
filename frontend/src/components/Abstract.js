import React from "react";

function Abstract({generatedAbstract, onClose}) {
    return (
        <div className="abstract__overlay">
            <div className="abstract__wrapper">
                <div className="abstract__wrapper__results">
                    <h2>Generated Abstract:</h2>
                    <hr/>
                    <p>{generatedAbstract}</p>
                </div>
                <button className="abstract__close" onClick={onClose}>
                    X
                </button>
            </div>
        </div>
    );
}

export default Abstract;
