import React, {useState} from "react";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";

function Search({onSearch, onToggleFilter}) {
    const [searchTerm, setSearchTerm] = useState("");

    const handleChange = (e) => {
        setSearchTerm(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!searchTerm.trim()) return;

        const response = await fetch('http://localhost:5001/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({query: searchTerm}),
        });

        const data = await response.json();
        onSearch(data);
    };

    return (
        <div className="search__wrapper">
            <div className="nav__button">
                <FontAwesomeIcon icon="fa-solid fa-bars" onClick={onToggleFilter}/>
            </div>
            <form onSubmit={handleSubmit} className="search__form">
                <input
                    type="text"
                    placeholder="Machine Learning..."
                    value={searchTerm}
                    onChange={handleChange}
                    className="search__input"
                />
                <button type="submit" className="search__button">Search</button>
            </form>
        </div>
    );
}

export default Search;
