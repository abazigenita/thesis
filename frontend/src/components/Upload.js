import React, {useState} from 'react';
import axios from 'axios';
import Form from './Form';

function Upload({onNewData, setLoading}) {
    const [showForm, setShowForm] = useState(false);

    const handleOpenForm = () => {
        setShowForm(true);
    };

    const handleCloseForm = () => {
        setShowForm(false);
    };

    const handleSubmit = async (formData) => {
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:5002/submit-paper', formData);
            if (response.status === 200) {
                const responseData = response.data;
                const newDataPoint = {
                    ...responseData, // Contains new paper data for the map
                    similarPapers: responseData.similar_papers // Contains similar papers' details
                };
                // Update the parent component with the new paper data
                onNewData(newDataPoint);
                handleCloseForm(); // Close the form after submission
            } else {
                console.error('Error submitting paper:', response.status);
            }
        } catch (error) {
            console.error('Error submitting paper:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <div className="upload__wrapper">
                <button onClick={handleOpenForm} className="upload__button">Upload your abstract</button>
            </div>
            {showForm && <Form onClose={handleCloseForm} onSubmit={handleSubmit}/>}
        </>
    );
}

export default Upload;
