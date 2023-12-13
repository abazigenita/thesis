import React, {useState, useCallback} from "react";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";

function Search({onSearch, onToggleFilter, setLoading, showFilter}) {
    const [searchTerm, setSearchTerm] = useState("");

    const handleChange = useCallback((e) => {
        setSearchTerm(e.target.value);
    }, []);

    const handleSubmit = useCallback(async (e) => {
        e.preventDefault();

        setLoading(true);

        const response = await fetch('http://localhost:5002/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: searchTerm}),
        });

        const data = await response.json();
        onSearch(data);

        setLoading(false);
    }, [onSearch, searchTerm, setLoading]);

    return (
        <div className="search__wrapper">
            <div className="nav__button">
                <FontAwesomeIcon
                    icon={showFilter ? "fa-solid fa-x" : "fa-solid fa-bars"}
                    onClick={onToggleFilter}
                />
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
