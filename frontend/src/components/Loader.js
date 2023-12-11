import React from 'react';

function Loader({isLoading}) {
    return (
        <div className={`loader-wrapper ${isLoading ? '' : 'hidden'}`}>
            <div className="loader"></div>
        </div>
    );
}

export default Loader;
