import React from 'react';

function SearchResults({results}) {
    return (
        <div className="search-results">
            {results.map((abstract, index) => (
                <p key={index} className="search-results__result">
                    {abstract}
                </p>
            ))}
        </div>
    );
}

export default SearchResults;
