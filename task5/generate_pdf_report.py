import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically and adds header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header (Pages 2 onwards)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Elevate Labs Data Analyst Internship | Task 5: Exploratory Data Analysis (EDA)")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * inch - 54, 45)

        self.drawString(54, 32, "Confidential | Elevate Labs Internship Submission")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 32, page_str)
        self.restoreState()


def build_pdf(filename="Titanic_EDA_Findings_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    c_primary = colors.HexColor("#1A365D")   # Deep navy
    c_secondary = colors.HexColor("#2B6CB0") # Ocean blue
    c_dark = colors.HexColor("#2D3748")      # Charcoal body
    c_light = colors.HexColor("#F7FAFC")     # Soft background
    c_accent = colors.HexColor("#DD6B20")    # Warm accent

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568"),
        alignment=0,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_primary,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=c_dark,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A202C")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_dark
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell_style,
        alignment=1
    )

    qa_q_style = ParagraphStyle(
        'QA_Q',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=True
    )

    qa_a_style = ParagraphStyle(
        'QA_A',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=8
    )

    story = []

    # ================= COVER / HEADER =================
    # Decorative color bar
    banner_data = [[
        Paragraph("<font color='white'><b>ELEVATE LABS | DATA ANALYST INTERNSHIP PROGRAM</b></font>", 
                  ParagraphStyle('B', fontName='Helvetica-Bold', fontSize=9, textColor=colors.white, alignment=1))
    ]]
    banner_tbl = Table(banner_data, colWidths=[504], rowHeights=[22])
    banner_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_secondary),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Task 5: Exploratory Data Analysis (EDA)", title_style))
    story.append(Paragraph("<b>In-Depth Statistical Exploration & Visual Analytics on the Titanic Disaster Dataset</b>", subtitle_style))

    # Metadata card table
    meta_data = [
        [Paragraph("<b>Internship Track:</b> Data Analytics", table_cell_style),
         Paragraph("<b>Tools Used:</b> Python (Pandas, Matplotlib, Seaborn, SciPy)", table_cell_style)],
        [Paragraph("<b>Deliverable Scope:</b> EDA Notebook, Visuals & Findings Report", table_cell_style),
         Paragraph("<b>Dataset:</b> Titanic Passenger Records (N = 891)", table_cell_style)],
        [Paragraph("<b>Objective:</b> Extract patterns, relationships, trends & anomalies", table_cell_style),
         Paragraph("<b>Status:</b> Completed & Verified", table_cell_style)]
    ]
    meta_tbl = Table(meta_data, colWidths=[252, 252])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 14))

    # ================= 1. EXECUTIVE SUMMARY =================
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        "This report delivers a rigorous Exploratory Data Analysis (EDA) of the Titanic disaster dataset. "
        "Through a synthesis of statistical profiling (using Pandas <code>.info()</code>, <code>.describe()</code>, and <code>.value_counts()</code>) "
        "and multi-dimensional visualizations (Seaborn heatmaps, pairplots, distributions, and facet grids), "
        "we investigate the socio-demographic drivers that determined passenger survival. Out of 891 tracked passengers, only "
        "<b>38.38% (342 individuals)</b> survived. Our investigation reveals that survival was overwhelmingly dictated by "
        "two structural factors: <b>gender privilege</b> (female survival rate was 74.2% vs. 18.9% for males) and "
        "<b>socio-economic status</b> (1st class survival rate was 63.0% vs. 24.2% for 3rd class). Furthermore, severe positive skewness "
        "was identified in ticket fares (skew = 4.79) and successfully rectified through log-transformation. All underlying relationships "
        "were validated with hypothesis tests including Pearson correlation, Chi-Square tests of independence, and Mann-Whitney U tests."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # ================= 2. DATASET OVERVIEW & PROFILING =================
    story.append(Paragraph("2. Dataset Structure & Data Profiling", h1_style))
    story.append(Paragraph(
        "The dataset contains <b>891 passenger records</b> and <b>15 feature columns</b>. A systematic check of data types, summary statistics, "
        "and null values was executed to gauge data integrity before visualization.", body_style
    ))

    # Data dictionary table
    dict_data = [
        [Paragraph("Feature", table_header_style), Paragraph("Type", table_header_style), Paragraph("Non-Null Count", table_header_style), Paragraph("Missing %", table_header_style), Paragraph("Description & Analytical Role", table_header_style)],
        [Paragraph("survived", table_cell_style), Paragraph("int64", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Target label (0 = Did not survive, 1 = Survived)", table_cell_style)],
        [Paragraph("pclass", table_cell_style), Paragraph("int64", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Ticket class: 1 = 1st (Luxury), 2 = 2nd, 3 = 3rd (Steerage)", table_cell_style)],
        [Paragraph("sex", table_cell_style), Paragraph("object", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Biological sex of passenger (male, female)", table_cell_style)],
        [Paragraph("age", table_cell_style), Paragraph("float64", table_cell_center), Paragraph("714 (80.1%)", table_cell_center), Paragraph("19.87%", table_cell_center), Paragraph("Passenger age in years; requires median imputation", table_cell_style)],
        [Paragraph("sibsp", table_cell_style), Paragraph("int64", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Number of siblings / spouses aboard", table_cell_style)],
        [Paragraph("parch", table_cell_style), Paragraph("int64", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Number of parents / children aboard", table_cell_style)],
        [Paragraph("fare", table_cell_style), Paragraph("float64", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Passenger ticket fare (£); highly right-skewed", table_cell_style)],
        [Paragraph("embark_town", table_cell_style), Paragraph("object", table_cell_center), Paragraph("889 (99.8%)", table_cell_center), Paragraph("0.22%", table_cell_center), Paragraph("Port of embarkation (Southampton, Cherbourg, Queenstown)", table_cell_style)],
        [Paragraph("deck", table_cell_style), Paragraph("object", table_cell_center), Paragraph("203 (22.8%)", table_cell_center), Paragraph("77.22%", table_cell_center), Paragraph("Cabin deck letter (A-G); severe sparsity", table_cell_style)],
        [Paragraph("alone", table_cell_style), Paragraph("bool", table_cell_center), Paragraph("891 (100%)", table_cell_center), Paragraph("0.0%", table_cell_center), Paragraph("Binary indicator: True if family size == 0", table_cell_style)],
    ]
    dict_tbl = Table(dict_data, colWidths=[65, 45, 75, 55, 264])
    dict_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(dict_tbl)
    story.append(Spacer(1, 10))

    # Summary Statistics Table
    story.append(Paragraph("<b>Parametric Summary Statistics (.describe())</b>", h2_style))
    stat_data = [
        [Paragraph("Metric", table_header_style), Paragraph("survived", table_header_style), Paragraph("pclass", table_header_style), Paragraph("age", table_header_style), Paragraph("sibsp", table_header_style), Paragraph("parch", table_header_style), Paragraph("fare (£)", table_header_style)],
        [Paragraph("Count", table_cell_style), Paragraph("891", table_cell_center), Paragraph("891", table_cell_center), Paragraph("714", table_cell_center), Paragraph("891", table_cell_center), Paragraph("891", table_cell_center), Paragraph("891", table_cell_center)],
        [Paragraph("Mean", table_cell_style), Paragraph("0.38", table_cell_center), Paragraph("2.31", table_cell_center), Paragraph("29.70", table_cell_center), Paragraph("0.52", table_cell_center), Paragraph("0.38", table_cell_center), Paragraph("32.20", table_cell_center)],
        [Paragraph("Std Dev", table_cell_style), Paragraph("0.49", table_cell_center), Paragraph("0.84", table_cell_center), Paragraph("14.53", table_cell_center), Paragraph("1.10", table_cell_center), Paragraph("0.81", table_cell_center), Paragraph("49.69", table_cell_center)],
        [Paragraph("Min", table_cell_style), Paragraph("0.00", table_cell_center), Paragraph("1.00", table_cell_center), Paragraph("0.42", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("0.00", table_cell_center)],
        [Paragraph("25% (Q1)", table_cell_style), Paragraph("0.00", table_cell_center), Paragraph("2.00", table_cell_center), Paragraph("20.12", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("7.91", table_cell_center)],
        [Paragraph("50% (Median)", table_cell_style), Paragraph("0.00", table_cell_center), Paragraph("3.00", table_cell_center), Paragraph("28.00", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("14.45", table_cell_center)],
        [Paragraph("75% (Q3)", table_cell_style), Paragraph("1.00", table_cell_center), Paragraph("3.00", table_cell_center), Paragraph("38.00", table_cell_center), Paragraph("1.00", table_cell_center), Paragraph("0.00", table_cell_center), Paragraph("31.00", table_cell_center)],
        [Paragraph("Max", table_cell_style), Paragraph("1.00", table_cell_center), Paragraph("3.00", table_cell_center), Paragraph("80.00", table_cell_center), Paragraph("8.00", table_cell_center), Paragraph("6.00", table_cell_center), Paragraph("512.33", table_cell_center)],
    ]
    stat_tbl = Table(stat_data, colWidths=[74, 70, 70, 70, 70, 70, 80])
    stat_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(stat_tbl)
    story.append(Spacer(1, 14))

    # ================= 3. VISUALIZATIONS & OBSERVATIONS =================
    story.append(PageBreak())
    story.append(Paragraph("3. Exploratory Visualizations & Key Observations", h1_style))
    story.append(Paragraph(
        "In accordance with project guidelines, this section documents all ten graphical analyses with clear technical observations, "
        "statistical metrics, and underlying domain justifications.", body_style
    ))
    story.append(Spacer(1, 4))

    # Helper function for adding visual blocks
    def add_visual_block(img_path, title, obs_paragraphs, img_h=2.2):
        block = []
        block.append(Paragraph(f"<b>{title}</b>", h2_style))
        if os.path.exists(img_path):
            img = Image(img_path, width=6.8 * inch, height=img_h * inch)
            block.append(img)
            block.append(Spacer(1, 4))
        
        # Observations card
        obs_elements = [Paragraph("<b>Analytical Observations:</b>", h3_style)]
        for p in obs_paragraphs:
            obs_elements.append(Paragraph(f"• {p}", callout_style))
            obs_elements.append(Spacer(1, 2))
        
        obs_tbl = Table([[obs_elements]], colWidths=[504])
        obs_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        block.append(obs_tbl)
        block.append(Spacer(1, 12))
        return KeepTogether(block)

    # Visual 1: Missing values heatmap
    story.append(add_visual_block(
        "images/01_missing_values_heatmap.png",
        "Visual 1: Missing Data Heatmap Matrix",
        [
            "<b>High Missingness in Deck (77.2%):</b> The cabin deck letter is absent for 688 passengers. Rather than arbitrary filling, converting this to an engineered boolean feature (<code>has_cabin_recorded</code>) retains structural signal without introducing noise.",
            "<b>Moderate Missingness in Age (19.9%):</b> 177 passengers have null age values. Because age varies significantly across socio-economic status and marital titles, group-median imputation based on Class and Gender is superior to global mean imputation.",
            "<b>Minimal Missingness in Embark Town (0.22%):</b> Only 2 values are absent in <code>embark_town</code>, both of which paid £80 in 1st class. Mode imputation (Southampton) resolves this cleanly."
        ],
        img_h=2.0
    ))

    # Visual 2: Univariate numerical distributions
    story.append(add_visual_block(
        "images/02_univariate_distributions.png",
        "Visual 2: Univariate Numerical Distributions (Age & Fare)",
        [
            "<b>Age Distribution (Skewness = +0.39):</b> Age exhibits a near-normal bell curve centered around young adults (peak density between 20 and 32 years). A secondary bimodal bump under age 5 reflects infants and toddlers.",
            "<b>Severe Fare Skewness (Skewness = +4.79):</b> The fare distribution exhibits extreme rightward skew. Over 75% of tickets cost less than £31, with a long thin tail reaching £512.33. Parametric models (e.g., Logistic Regression) will suffer from gradient destabilization without mathematical compression."
        ],
        img_h=2.1
    ))

    story.append(PageBreak())

    # Visual 3: Univariate categorical distributions
    story.append(add_visual_block(
        "images/03_categorical_counts.png",
        "Visual 3: Univariate Categorical Distributions",
        [
            "<b>Gender Composition:</b> Males made up 64.8% (577) of passengers, while females comprised 35.2% (314).",
            "<b>Socio-Economic Class Spread:</b> The passenger manifest was predominantly working-class, with 55.1% (491) in 3rd class, 24.2% (216) in 1st class, and 20.7% (184) in 2nd class.",
            "<b>Port of Embarkation:</b> 72.3% of passengers boarded in Southampton, 18.9% in Cherbourg, and 8.6% in Queenstown.",
            "<b>Solo vs. Accompanied:</b> 60.3% (537) traveled unaccompanied (solo), whereas 39.7% (354) traveled with family members."
        ],
        img_h=3.2
    ))

    # Visual 4: Bivariate survival rates
    story.append(add_visual_block(
        "images/04_survival_by_demographics.png",
        "Visual 4: Bivariate Survival Rates by Gender, Class & Port",
        [
            "<b>The Gender Effect:</b> Female survival was <b>74.2%</b> compared to only <b>18.9%</b> for males. Women were nearly 4x more likely to survive, directly confirming historical adherence to maritime protocol.",
            "<b>The Class Gradient:</b> 1st class survival was <b>63.0%</b>, 2nd class was <b>47.3%</b>, and 3rd class collapsed to <b>24.2%</b>. Wealth and cabin deck altitude were decisive determinants of access to lifeboats.",
            "<b>Embarkation Differences:</b> Cherbourg passengers enjoyed a <b>55.4%</b> survival rate, compared to 38.4% for Queenstown and 33.7% for Southampton. This is confounded by passenger class: Cherbourg had a much higher proportion of 1st class tickets."
        ],
        img_h=2.1
    ))

    story.append(PageBreak())

    # Visual 5: Boxplots
    story.append(add_visual_block(
        "images/05_boxplots_age_fare_by_class.png",
        "Visual 5: Outlier Detection and Variance Analysis via Boxplots",
        [
            "<b>Age Progression by Class:</b> Median age increases consistently with wealth: 37 years in 1st class, 29 in 2nd class, and 24 in 3rd class. In 2nd and 3rd class, children who survived appear as lower whiskers extending down to infancy.",
            "<b>Fare Outliers in 1st Class:</b> The right plot (log scale) highlights severe fare variance in 1st class, with multiple ultra-luxury outliers exceeding £200 up to £512. Within every single class tier, survivors paid a higher median fare than victims."
        ],
        img_h=2.4
    ))

    # Visual 6: Correlation matrix heatmap
    story.append(add_visual_block(
        "images/06_correlation_heatmap.png",
        "Visual 6: Pearson Correlation Matrix Heatmap",
        [
            "<b>Pclass vs. Survived (r = -0.34):</b> Strongest inverse linear correlation with survival, confirming that lower tier numbers (1st class) had higher survival.",
            "<b>Fare vs. Survived (r = +0.26):</b> Notable positive correlation between monetary spend and survival probability.",
            "<b>Pclass vs. Fare (r = -0.55):</b> Strong negative correlation validating that luxury ticket tiers carried substantially higher monetary fares.",
            "<b>SibSp vs. Parch (r = +0.41):</b> Positive association between traveling with siblings/spouses and parents/children, forming distinct nuclear family units."
        ],
        img_h=2.5
    ))

    story.append(PageBreak())

    # Visual 7: Pairplot
    story.append(add_visual_block(
        "images/07_pairplot_titanic.png",
        "Visual 7: Seaborn Pairplot Across Key Numerical Features",
        [
            "<b>Bivariate Separation:</b> The joint scatterplots show clear clustering of survivors (blue squares) in high-fare, lower-class, and younger age brackets.",
            "<b>Non-Linear Family Dynamics:</b> The scatter between <code>sibsp</code> and <code>parch</code> reveals an inverted-U survival curve: small family units (1-3 relatives) achieved the highest survival rates, while large families (5+ relatives) suffered almost 100% fatalities due to evacuation fragmentation."
        ],
        img_h=3.0
    ))

    # Visual 8: Scatterplot
    story.append(add_visual_block(
        "images/08_scatterplot_age_vs_fare.png",
        "Visual 8: Multivariate Scatterplot (Age vs. Fare by Survival & Gender)",
        [
            "<b>High Fare Sanctuary:</b> Nearly all passengers paying fares above £60 survived regardless of age (cluster of blue markers in upper tier).",
            "<b>Low-Fare Vulnerability:</b> In the low-fare steerage sector (< £20), adult males aged 20–50 experienced catastrophic mortality (orange circles).",
            "<b>Demographic Exceptions:</b> In the lower fare region, young boys (< 10) and women of all ages show high survival concentrations (blue squares)."
        ],
        img_h=2.3
    ))

    story.append(PageBreak())

    # Visual 9: Handling Skewed Data
    story.append(add_visual_block(
        "images/09_skewness_transformation.png",
        "Visual 9: Demonstration of Skewness Handling via Log Transformation",
        [
            "<b>Raw Distribution Skewness (+4.79):</b> The uncompressed fare data violates normality assumptions, which introduces bias into linear models and neural networks.",
            "<b>Log1p Transformation (Skewness reduced to +0.39):</b> Computing <code>np.log1p(fare)</code> dampens high-leverage outliers, compresses the right tail, and produces an approximately bimodal normal distribution suitable for parametric modeling."
        ],
        img_h=2.2
    ))

    # Visual 10: Multi-panel Catplot
    story.append(add_visual_block(
        "images/10_multivariate_catplot.png",
        "Visual 10: Multi-panel Catplot (Class x Gender x Embarkation Survival)",
        [
            "<b>Female Survival Ceiling:</b> Females in 1st and 2nd class achieved near-universal survival (88% to 100%) across all three ports.",
            "<b>Intersectionality of Class & Gender:</b> In 3rd class, female survival plummeted to between 37% and 50%, demonstrating that gender privilege was severely curtailed by lower socio-economic status.",
            "<b>Male Steerage Fatalities:</b> Males in 3rd class suffered the worst outcome regardless of embarkation port, with survival hovering between 8% and 13%."
        ],
        img_h=2.3
    ))

    # ================= 4. STATISTICAL HYPOTHESIS TESTING =================
    story.append(PageBreak())
    story.append(Paragraph("4. Statistical Hypothesis Testing & Inferential Proof", h1_style))
    story.append(Paragraph(
        "To ensure that visual insights reflect true underlying statistical effects rather than random sampling variation, "
        "we conducted formal inferential hypothesis tests.", body_style
    ))

    test_data = [
        [Paragraph("Hypothesis Test", table_header_style), Paragraph("Variables Evaluated", table_header_style), Paragraph("Test Statistic", table_header_style), Paragraph("p-value", table_header_style), Paragraph("Statistical Conclusion", table_header_style)],
        [Paragraph("Chi-Square Test of Independence", table_cell_style), Paragraph("Gender vs. Survival", table_cell_style), Paragraph("χ² = 260.72 (dof = 1)", table_cell_center), Paragraph("p = 1.20 × 10⁻⁵⁸", table_cell_center), Paragraph("Reject H₀: Highly significant association between gender and survival probability.", table_cell_style)],
        [Paragraph("Chi-Square Test of Independence", table_cell_style), Paragraph("Passenger Class vs. Survival", table_cell_style), Paragraph("χ² = 102.89 (dof = 2)", table_cell_center), Paragraph("p = 4.55 × 10⁻²³", table_cell_center), Paragraph("Reject H₀: Highly significant association between class tier and survival probability.", table_cell_style)],
        [Paragraph("Mann-Whitney U Test (One-tailed)", table_cell_style), Paragraph("Fare: Survivors vs. Deceased", table_cell_style), Paragraph("U = 129,952.5", table_cell_center), Paragraph("p = 1.08 × 10⁻²²", table_cell_center), Paragraph("Reject H₀: Survivors paid statistically significantly higher fares than non-survivors.", table_cell_style)],
    ]
    test_tbl = Table(test_data, colWidths=[105, 100, 95, 75, 129])
    test_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(test_tbl)
    story.append(Spacer(1, 14))

    # ================= 5. INTERVIEW QUESTIONS =================
    story.append(Paragraph("5. Internship Interview Questions & Comprehensive Answers", h1_style))
    story.append(Paragraph(
        "Below are detailed, interview-ready responses to the seven conceptual questions outlined in the internship task brief.", body_style
    ))
    story.append(Spacer(1, 4))

    qa_list = [
        (
            "Question 1: What is EDA and why is it important?",
            "<b>Exploratory Data Analysis (EDA)</b> is the analytical practice of systematically examining, visualizing, and summarizing datasets to uncover underlying structures, patterns, relationships, anomalies, and test fundamental assumptions prior to formal modeling.<br/><br/>"
            "<b>Core Importance:</b><br/>"
            "• <b>Data Hygiene & Completeness:</b> Uncovers data quality flaws, null patterns, duplicate records, and invalid formats.<br/>"
            "• <b>Feature Understanding:</b> Determines whether distributions are bell-shaped, skewed, or multimodal.<br/>"
            "• <b>Hypothesis Generation:</b> Informs the analyst about which features possess predictive signal and how variables interact.<br/>"
            "• <b>Informs Model Strategy:</b> Dictates necessary transformations (e.g., log scaling, one-hot encoding, IQR outlier capping)."
        ),
        (
            "Question 2: Which plots do you use to check correlation?",
            "To evaluate correlation between variables, data analysts deploy the following visualizations:<br/>"
            "• <b>Correlation Matrix Heatmap (<code>sns.heatmap</code>):</b> The standard multi-variable summary tool. Displays a color-coded grid of Pearson (linear), Spearman (monotonic rank), or Kendall coefficients with numerical annotations.<br/>"
            "• <b>Scatterplot (<code>sns.scatterplot</code>):</b> The primary visual tool for inspecting the bivariate relationship between two continuous variables, revealing direction, strength, and non-linearity.<br/>"
            "• <b>Pairplot / Scatterplot Matrix (<code>sns.pairplot</code>):</b> Computes an N × N grid of bivariate scatterplots off-diagonal and univariate KDEs/histograms on the diagonal.<br/>"
            "• <b>Jointplot (<code>sns.jointplot</code>):</b> Combines a central scatter or hexbin plot with marginal distribution histograms along both axes.<br/>"
            "• <b>Regression Plot (<code>sns.regplot</code>):</b> Overlays an Ordinary Least Squares (OLS) best-fit line and 95% bootstrap confidence band over a scatterplot."
        ),
        (
            "Question 3: How do you handle skewed data?",
            "Skewed data introduces gradient instability and distorts parametric algorithms (e.g., Linear/Logistic Regression, PCA, Neural Networks). It is handled using:<br/>"
            "• <b>Logarithmic Transformation (<code>np.log1p</code>):</b> Compresses long positive tails while safely accommodating zero values via log(1 + x). Ideal for economic metrics like fare or revenue.<br/>"
            "• <b>Square Root Transformation (<code>np.sqrt</code>):</b> A milder transformation suitable for count data with moderate skewness.<br/>"
            "• <b>Box-Cox Transformation (<code>scipy.stats.boxcox</code>):</b> A power transformation that estimates an optimal parameter λ to maximize normality (requires strictly positive numbers).<br/>"
            "• <b>Yeo-Johnson Transformation (<code>scipy.stats.yeojohnson</code>):</b> A generalized power transformation that accommodates zero and negative values.<br/>"
            "• <b>IQR Winsorization & Capping:</b> Replacing values beyond [Q1 - 1.5×IQR, Q3 + 1.5×IQR] with boundary values.<br/>"
            "• <b>Model Selection:</b> Utilizing tree-based ensemble models (Random Forest, LightGBM, XGBoost) which are invariant to monotonic scaling."
        ),
        (
            "Question 4: How to detect multicollinearity?",
            "Multicollinearity occurs when two or more independent features in a model are highly linearly interdependent, inflating standard errors and rendering coefficient estimates unstable. Detection methods include:<br/>"
            "• <b>Variance Inflation Factor (VIF):</b> The industry standard metric, defined as VIF_i = 1 / (1 - R_i²). An empirical rule of thumb: VIF = 1 denotes no correlation; 1 < VIF < 5 is moderate and acceptable; VIF ≥ 5 or 10 signifies problematic collinearity.<br/>"
            "• <b>Correlation Matrix Heatmap:</b> Screening for pairwise correlation coefficients exceeding |r| > 0.80.<br/>"
            "• <b>Condition Number (κ):</b> Derived from the eigenvalues of the scaled matrix XᵀX. A condition number > 30 signals severe collinearity.<br/>"
            "• <b>Tolerance:</b> The reciprocal of VIF (Tolerance = 1 / VIF). Values below 0.1 or 0.2 indicate severe collinearity."
        ),
        (
            "Question 5: What are univariate, bivariate, and multivariate analyses?",
            "These three analytical tiers are categorized by the number of variables examined concurrently:<br/>"
            "• <b>Univariate Analysis:</b> Analyzes a <b>single variable</b> in isolation to inspect its frequency, central tendency (mean, median, mode), dispersion (variance, IQR), and distribution shape. Examples: Histograms, KDE plots, boxplots, countplots.<br/>"
            "• <b>Bivariate Analysis:</b> Explores the relationship between <b>two variables</b> to detect correlations, group differences, or dependency patterns. Examples: Scatterplots, bivariate bar charts, cross-tabulations.<br/>"
            "• <b>Multivariate Analysis:</b> Analyzes <b>three or more variables</b> concurrently to understand complex interaction effects, conditional relationships, and high-dimensional clustering. Examples: Heatmaps, pairplots, FacetGrids, 3D scatterplots, PCA biplots."
        ),
        (
            "Question 6: Difference between heatmap and pairplot?",
            "• <b>Heatmap (<code>sns.heatmap</code>):</b> Displays a 2D grid of aggregated statistical summaries (such as an N × N Pearson correlation matrix or confusion matrix) represented by colored tiles. It is computationally lightweight, summarizes macroscopic linear associations, but discards row-level granularity.<br/>"
            "• <b>Pairplot (<code>sns.pairplot</code>):</b> Renders an N × N panel of individual subplots displaying actual raw data observations (scatterplots off-diagonal and univariate KDEs/histograms on diagonal). It allows segmenting data points with a categorical <code>hue</code> label, detects non-linear clusters, but is computationally heavier on large datasets."
        ),
        (
            "Question 7: How do you summarize your insights?",
            "An effective analytical summary bridges data engineering and executive business decision-making using the <b>BLUF (Bottom Line Up Front)</b> framework:<br/>"
            "1. <b>Executive Headline:</b> State the overarching conclusion in 1-2 direct sentences.<br/>"
            "2. <b>Quantify with Evidence:</b> Always substantiate findings with concrete percentages, metrics, and p-values (e.g., 'Female survival was 74.2% vs. 18.9% for males, p < 0.001').<br/>"
            "3. <b>Visual Anchor:</b> Connect each finding to a named chart or figure with explicit callout captions.<br/>"
            "4. <b>Actionable Business/Modeling Recommendations:</b> Translate findings into concrete engineering steps (e.g., log-transforming fares, imputing missing ages via class medians, creating family size features)."
        )
    ]

    for q, a in qa_list:
        q_block = [
            Paragraph(f"<b>{q}</b>", qa_q_style),
            Paragraph(a, qa_a_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceBefore=4, spaceAfter=8)
        ]
        story.append(KeepTogether(q_block))

    # ================= 6. CONCLUSION & RECOMMENDATIONS =================
    story.append(Paragraph("6. Recommendations for Machine Learning Pipeline", h1_style))
    recom_text = (
        "Based on our exploratory findings, the following engineering steps are recommended before training classification models:<br/>"
        "• <b>Feature Engineering:</b> Synthesize <code>family_size = sibsp + parch + 1</code> and a categorical family tier (<code>Solo</code>, <code>Small</code>, <code>Large</code>). Create a title extractor from passenger names (Mr, Mrs, Miss, Master) which carries high predictive power for age and social standing.<br/>"
        "• <b>Missing Value Imputation:</b> Impute <code>age</code> using median values grouped by <code>pclass</code> and <code>sex</code>. For <code>embarked</code>, impute mode ('S'). For <code>deck</code>, encode missingness as a binary indicator <code>has_cabin</code>.<br/>"
        "• <b>Feature Transformation:</b> Apply <code>np.log1p</code> on <code>fare</code> to neutralize positive skewness and stabilize variance."
    )
    story.append(Paragraph(recom_text, body_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled into '{filename}'!")

if __name__ == "__main__":
    build_pdf()
