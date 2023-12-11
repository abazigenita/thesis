import React, {useState, useMemo} from "react";
import axios from "axios";

function Filter({onFilter}) {
    const [filters, setFilters] = useState({
        category: "",
        subcategories: [],
        dateGroup: "Any time",
        onlyLicensed: false,
    });

    const categories = useMemo(() => [
        "Physics",
        "Mathematics",
        "Computer Science",
        "Quantitative Biology",
        "Quantitative Finance",
        "Statistics",
        "Electrical Engineering and Systems Science",
        "Economics",
    ], []);

    const subcategories = useMemo(() => ({
        Physics: [
            "astro-ph",
            "cond-mat",
            "gr-qc",
            "hep-ex",
            "hep-lat",
            "hep-ph",
            "hep-th",
            "math-ph",
            "nlin",
            "nucl-ex",
            "nucl-th",
            "physics",
            "quant-ph",
        ],
        Mathematics: ["math"],
        "Computer Science": ["CoRR"],
        "Quantitative Biology": ["q-bio"],
        "Quantitative Finance": ["q-fin"],
        Statistics: ["stat"],
        "Electrical Engineering and Systems Science": ["eess"],
        Economics: ["econ"],
    }), []);

    const dateGroups = ["Any time", "Since 2023", "Since 2022", "Since 2019"];

    const handleCategoryChange = async (event) => {
        const selectedCategory = event.target.value;
        const selectedSubcategories = subcategories[selectedCategory] || [];

        setFilters({
            ...filters,
            category: selectedCategory,
            subcategories: selectedSubcategories,
        });

        await applyFilter("category", selectedCategory, selectedSubcategories);
    };

    const handleDateGroupChange = async (dateGroup) => {
        setFilters({...filters, dateGroup});
        await applyFilter("date", dateGroup);
    };


    const handleLicenseChange = async () => {
        const updatedLicenseState = !filters.onlyLicensed;
        setFilters({...filters, onlyLicensed: updatedLicenseState});
        await applyFilter("license", updatedLicenseState);
    };

    const applyFilter = async (filterType, value) => {
        let apiUrl, postData;
        switch (filterType) {
            case "category":
                apiUrl = "http://localhost:5001/filter-category";
                postData = {categories: [value, ...subcategories[value]]};
                break;
            case "date":
                apiUrl = "http://localhost:5001/filter-date";
                postData = {dateGroup: value};
                break;
            case "license":
                apiUrl = "http://localhost:5001/filter-license";
                postData = {onlyLicensed: value};
                break;
            default:
                return;
        }

        try {
            const response = await axios.post(apiUrl, postData);
            onFilter(response.data); // Update the map data with filtered results
        } catch (error) {
            console.error("Error while applying filter:", error);
        }
    };

    return (
        <div className="filter__wrapper">
            <form className="filter__form">
                <div className="filter__wrapper__div">
                    {/* Category filter */}
                    <label>Category:</label>
                    <select
                        name="category"
                        value={filters.category}
                        onChange={handleCategoryChange}
                        className="filter__category"
                    >
                        <option value="">Select a category</option>
                        {categories.map((category, index) => (
                            <option key={index} value={category}>
                                {category}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="filter__wrapper__div">
                    {/* Date group filter */}
                    <div className="filter__date__group">
                        {dateGroups.map((group, index) => (
                            <p
                                key={index}
                                className={`date__group-item ${
                                    filters.dateGroup === group ? "selected" : ""
                                }`}
                                onClick={() => handleDateGroupChange(group)}
                            >
                                {group}
                            </p>
                        ))}
                    </div>
                </div>
                <div className="filter__wrapper__div">
                    {/* License filter */}
                    <label>
                        <input
                            type="checkbox"
                            name="onlyLicensed"
                            checked={filters.onlyLicensed}
                            onChange={handleLicenseChange}
                            className="filter__license"
                        />
                        Display only licensed papers
                    </label>
                </div>
            </form>
        </div>
    );
}

export default Filter;
