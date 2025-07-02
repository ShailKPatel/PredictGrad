## Documentation for EDA of student_performance_dataset

This documentation details the Exploratory Data Analysis (EDA) performed on the `student_performance_dataset`. The purpose of this EDA is to understand the distribution of these categorical and numerical variables, providing insights into the dataset's composition to inform potential machine learning tasks.

### Bar Plots for Distribution of Categorical Variables

This section uses bar plots to analyze the distribution of the categorical columns—Branch, Gender, and Religion—offering a visual summary of the dataset's composition.

#### Bar Plot: Distribution of Branch
The bar plot represents the count of students across various academic branches.
- **Categories**: CE, CSE, CST, IT, CS&IT, AIDS, CEA, AIML, CSD, RAI.
- **Findings**:
  - CE: 202 students (22.32% of total).
  - CSE: 198 students (21.88%).
  - CST: 172 students (19.01%).
  - IT: 136 students (15.03%).
  - CS&IT: 62 students (6.85%).
  - AIDS: 44 students (4.86%).
  - CEA: 42 students (4.64%).
  - AIML: 24 students (2.65%).
  - CSD: 13 students (1.44%).
  - RAI: 12 students (1.33%).
  - **Total Count**: 905 students.

#### Bar Plot: Distribution of Gender
The bar plot illustrates the count of students by gender.
- **Categories**: M (Male), F (Female).
- **Findings**:
  - Male: 733 students (80.99% of total).
  - Female: 172 students (19.01%).
  - **Total Count**: 905 students.

#### Bar Plot: Distribution of Religion
The bar plot depicts the count of students by religion.
- **Categories**: Hindu, Muslim, Jain, Christian, Sikh.
- **Findings**:
  - Hindu: 830 students (91.71% of total).
  - Muslim: 44 students (4.86%).
  - Jain: 27 students (2.98%).
  - Christian: 3 students (0.33%).
  - Sikh: 1 student (0.11%).
  - **Total Count**: 905 students.

### Histograms for Distribution of Numerical Variables

This section analyzes the distribution of numerical variables—focusing on theory marks, practical marks, and attendance—using histograms to provide a visual summary of the dataset's composition.
#### Histogram: Distribution of Theory Marks
The histogram represents the distribution of student scores in most theory exams.
- **Findings**:
  - The distribution of most theory marks is approximately normal, with a peak around the middle range of scores.
  - There is a slight left skew, with fewer students scoring very low and a small number achieving very high scores.

#### Histogram: Distribution of Practical Marks
The histogram depicts the distribution of student scores in most practical exams.
- **Findings**:
  - The distribution of most practical marks is left-skewed, with a peak toward the higher end of the score range.
  - Most students score higher marks, with fewer students scoring in the lower range, indicating that achieving higher scores in practical exams may be easier compared to theory exams.

#### Histogram: Distribution of Attendance
The histogram illustrates the distribution of student attendance percentages.
- **Findings**:
  - The distribution of attendance is highly left-skewed, with a significant peak at the higher end of the percentage range.
  - The majority of students have high attendance, likely influenced by a mandatory attendance policy or attendance-related bonus marks, with very few students having low attendance.