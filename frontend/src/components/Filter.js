import React, {useState, useMemo} from "react";
import axios from "axios";

function Filter({onFilter, onlyLicensed, setOnlyLicensed, setLoading}) {
    const [filters, setFilters] = useState({
        category: "",
        dateGroup: "Any time",
        onlyLicensed: onlyLicensed,
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

    const dateGroups = ["Any time", "Since 2023", "Since 2022", "Since 2019"];

    const handleCategoryChange = async (event) => {
        const selectedCategory = event.target.value;
        setFilters({...filters, category: selectedCategory});
        await applyFilter("category", selectedCategory);
    };

    const handleDateGroupChange = async (dateGroup) => {
        setFilters({...filters, dateGroup});
        await applyFilter("date", dateGroup);
    };

    const handleLicenseChange = async () => {
        const updatedLicenseState = !onlyLicensed;
        setOnlyLicensed(updatedLicenseState);
        await applyFilter("license", updatedLicenseState);
    };

    const applyFilter = async (filterType, value) => {
        setLoading(true);
        let apiUrl, postData;
        switch (filterType) {
            case "category":
                apiUrl = "http://backend:5002/filter-category";
                postData = {category: value};
                break;
            case "date":
                apiUrl = "http://backend:5002/filter-date";
                postData = {dateGroup: value};
                break;
            case "license":
                apiUrl = "http://backend:5002/filter-license";
                postData = {onlyLicensed: value};
                break;
            default:
                return;
        }

        try {
            const response = await axios.post(apiUrl, postData);
            onFilter(response.data);
            setLoading(false);
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
                        <option value="">All Categories</option>
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
                            checked={onlyLicensed}
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
