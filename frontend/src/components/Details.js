import React from "react";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";

function Details({selectedData, onClose}) {
    const isArxivId = (id) => {
        return /^\d+\.\d+$/.test(id);
    };

    return (
        <div className="details__wrapper">
            <div className="details__content">
                <h3 className="details__content__title">{selectedData.title}</h3>
                <p className="details__content__authors">Authors: <span>{selectedData.authors}</span></p>
                <span className="details__content__date">{selectedData.date}</span>
                <p className="details__content__abstract">{selectedData.abstract}</p>
                <div className="details__content__link">
                    {isArxivId(selectedData.id) &&
                        <a href={`https://arxiv.org/abs/0${selectedData.id}`}
                           target="_blank"
                           rel="noreferrer"
                        >
                            <FontAwesomeIcon icon="fa-solid fa-arrow-up-right-from-square"/> Paper Details
                        </a>
                    }
                </div>
            </div>
            <button className="details__wrapper__close" onClick={onClose}>
                X
            </button>
        </div>
    );
}

export default Details;
