import React, {useState, useEffect} from "react";

function Form({onClose, onSubmit}) {
    const [formData, setFormData] = useState({
        title: "",
        authors: "",
        abstract: "",
    });
    const [isFormValid, setIsFormValid] = useState(false);

    useEffect(() => {
        validateForm();
    }, [formData]);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value,
        });
    };

    const validateForm = () => {
        const isValid = formData.title && formData.authors && formData.abstract;
        setIsFormValid(isValid);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!isFormValid) return;

        onSubmit(formData);

        try {
            const response = await fetch("http://localhost:5001/submit-paper", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    title: formData.title,
                    authors: formData.authors,
                    abstract: formData.abstract,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log("Submission result:", result);
            onClose();
        } catch (error) {
            console.error("Error submitting paper:", error);
        }
    };

    return (
        <div className="upload-form__overlay">
            <div className="upload-form__wrapper">
                <form onSubmit={handleSubmit} className="upload-form__form">
                    <input
                        type="text"
                        name="title"
                        value={formData.title}
                        onChange={handleChange}
                        placeholder="Title"
                        className="upload-form__title"
                    />
                    <input
                        type="text"
                        name="authors"
                        value={formData.authors}
                        onChange={handleChange}
                        placeholder="Authors"
                        className="upload-form__authors"
                    />
                    <textarea
                        name="abstract"
                        value={formData.abstract}
                        onChange={handleChange}
                        placeholder="Abstract"
                        className="upload-form__abstract"
                    ></textarea>
                    <button
                        type="submit"
                        disabled={!isFormValid}
                        className={!isFormValid ? "button-disabled" : ""}
                    >
                        Submit
                    </button>
                </form>
                <button onClick={onClose} className="upload-form__close">
                    X
                </button>
            </div>
        </div>
    );
}

export default Form;
