# Football Data Visualization Project

**Repository:** \[GitHub Repository Link Here]

A suite of 13 interactive football visualizations built with Python and Plotly. This project transforms raw match and player data into actionable insights for coaches and analysts.

## Repository Structure

```
├── database.sqlite                # SQLite database with football data
├── football_presentation.html      # Static presentation of key findings
├── LICENSE                         # Project license
├── project.py                      # Main script to generate all visualizations
├── README.md                       # This file
├── sketches/                       # Preliminary design mockups (PDF)
└── visualizations/                 # Generated HTML visualization files
└── Project Report/                 # Project Report Docx

```

## Prerequisites

- Python 3.8+
- pip
- unzip (for decompressing the database)
- SQLite (optional, for manual inspection)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Unzip the database archive into the project root:

   ```bash
   unzip database.sqlite.zip -d .
   ```

3. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

4. Install required Python packages:

   ```bash
   pip install pandas numpy plotly
   ```

## Usage

1. Confirm `database.sqlite` is present in the project root (after unzipping).
2. Run the main script to generate visualizations:

   ```bash
   python project.py
   ```

3. Generated HTML files will appear in the `visualizations/` directory.
4. Open any `.html` file in your web browser to explore the interactive charts.

## Customization

- Adjust data filters, league selection, or default players by editing the `default_args` section in `project.py`.
- Change output directory or database path via the `OUTPUT_DIR` and `DB_PATH` variables at the top of `project.py`.

## Video Demonstration

A walk-through of the visualizations is available here:
\[https://www.youtube.com/watch?v=-jmFyWXQdVw&ab_channel=MuhammadShumayil]

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
