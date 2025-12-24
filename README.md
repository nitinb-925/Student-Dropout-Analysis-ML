# Predicting Student Dropout in Higher Education Using Machine Learning

**Objective**

The primary objective of this project is to investigate the key factors that contribute to student dropout in higher education and to build a predictive model that can identify students at risk of dropping out.
Using demographic characteristics, parental background, enrollment details, academic performance indicators, and macroeconomic context (unemployment, inflation, GDP), the project aims to:<br>
* Explore patterns and risk factors linked to higher dropout rates.
* Develop and evaluate classification models to distinguish:
  * Dropout
  * Enrolled
  * Graduate
* Provide a probabilistic risk score that can support early intervention and targeted retention strategies by universities.

**Dataset Description**

The dataset consists of 4,424 student records with 37 variables, each row representing one higher-education student and their progression status at the end of the normal course duration. All predictors are stored as numeric codes but correspond to rich categorical and continuous information. Demographic and enrollment variables include marital status (coded for single, married, widowed, divorced, etc.), application mode (different admission channels such as first-phase general contingent, over-23 entry, course transfer, international student, etc.), application order (0 = first choice through 9 = last choice), the degree course code (e.g., Management, Nursing, Journalism, Social Service, Informatics Engineering), daytime vs evening attendance (1 = daytime, 0 = evening), nationality, and an “international” flag indicating whether the student is an international enrollee. Prior education and family background are captured through previous qualification (e.g., secondary vs higher education), previous qualification grade (a continuous score), and detailed codes for the mother’s and father’s education levels (from basic schooling up to Bachelor, Master, Doctorate) and occupations (unskilled, skilled, technical, professional, managerial, etc.), which together approximate socio-economic status. Academic performance at the institution is summarized separately for the 1st and 2nd semesters via the number of curricular units credited, enrolled, evaluated, approved, average grade (0–20 scale), and the count of curricular units without evaluations in each semester, providing a detailed view of workload, assessment activity, success, and non-attendance. The dataset also includes administrative and support indicators such as whether the student is displaced, has educational special needs, is a debtor, whether tuition fees are up to date, gender, scholarship holder status, and age at enrollment, which jointly reflect both academic and financial vulnerability. Finally, each record is linked to macroeconomic context variables—unemployment rate, inflation rate, and GDP for the corresponding year—which provide a backdrop of labor-market and economic conditions that may influence continuation or dropout decisions. There are no missing values in any field, and the target variable is a three-class outcome (target) defining whether the student ultimately dropped out, remained enrolled, or graduated, making the problem a multi-class classification task.

**Exploratory Data Analysis**

<img width="902" height="252" alt="image" src="https://github.com/user-attachments/assets/29ee530c-f150-4252-a17e-6f1caf9ea4d3" />

                                                Figure 1

From the above figure 1, it can be observed that dropouts form a large portion of the student population, which suggests retention is a major issue. Single students [Martial Status = 1] are much more likely to drop out than students with other marital statuses — this could be due to lower family support or higher mobility. Understanding why single students drop out at higher rates can help design targeted support programs to improve retention.

<img width="902" height="301" alt="image" src="https://github.com/user-attachments/assets/a4ef422a-3dfc-46a6-ae10-ffd7d69ceec1" />

                                                Figure 2

From the above figure 2, it can be observed that the dropouts are heavily concentrated in a few key courses, particularly Management (evening attendance), Management, Nursing, and Journalism and Communication, each showing notably higher dropout counts than other programs. Courses related to business, communication, and health tend to have the highest dropout rates, while specialized programs like Biofuel Production Technologies and Oral Hygiene have the lowest. Portuguese students account for the vast majority of dropouts, making student retention primarily a domestic issue. Dropouts among international students are minimal, with only a handful of cases across nationalities such as Brazilian, Spanish, Cape Verdean, and a few others.

<img width="902" height="323" alt="image" src="https://github.com/user-attachments/assets/dc22f995-3190-4fd5-b02c-f1eddef2f3ea" />

                                               Figure 3

From the above figure 3, it can be observed that the gender does not appear to be a strong factor in dropout risk, but being a domestic student is clearly associated with a much higher dropout rate compared to international students.

<img width="902" height="329" alt="image" src="https://github.com/user-attachments/assets/bfb78bdb-25f6-47ce-8a2b-bae62af043d2" />

                                               Figure 4

From the above figure 4, it suggests that students transitioning directly from secondary education [Previous qualification = 1] or basic cycles are at higher risk of dropping out, highlighting the need for stronger support mechanisms and bridging programs for first-year university students who lock prior higher education experience.

From the below figure 5, it can be observed that the higher admission grades are positively associated with graduation, indicating that students with stronger academic preparation tend to succeed in completing their courses. Older age at enrollment is linked to a higher dropout risk, highlighting the need for flexible support systems for mature students who may balance studies with work or family responsibilities.

<img width="902" height="298" alt="image" src="https://github.com/user-attachments/assets/0fa0906a-f7ac-4b21-8a0b-d6f087f70957" />

                                                 Figure 5

From the below figure 6, it can be observed that the students whose parents have lower levels of education — particularly those with only Basic Education 1st cycle (4th/5th year) are much more likely to drop out. This is evident as the highest dropout counts are from students whose mothers and fathers have these lower schooling levels. Students whose parents completed Secondary Education (12th year) also show a significant number of dropouts, although the risk declines compared to parents with only basic education. Dropout numbers drop sharply among students whose parents hold Higher Education degrees — such as Bachelor’s, Master’s, or Doctorate. This suggests that higher parental education provides more support or advantages that help students persist to graduation.

<img width="902" height="285" alt="image" src="https://github.com/user-attachments/assets/f5b317f1-8857-4b88-83f8-bbdad5ab6202" />

                                                 Figure 6

From the below figure 7, it can be observed that the students whose parents work in lower-skilled, manual, or unskilled occupations have a significantly higher likelihood of dropping out. This trend is consistent for both mothers’ and fathers’ occupations, with unskilled labor being the most common background among dropout cases. Higher parental occupational status such as professional, technical, or managerial roles appears to be associated with lower student dropout rates. This suggests that economic stability and social capital linked to parental occupation may play an important role in student persistence.

<img width="902" height="274" alt="image" src="https://github.com/user-attachments/assets/4b380f27-3555-450d-87e5-b38651f3a3b9" />

                                                  Figure 7

**Data Preparation**

The dataset consisted of 4,424 student records and 37 variables covering demographics, parental background, academic history, course enrollment information, and socioeconomic indicators. An initial inspection confirmed that there were no missing values across any feature, enabling direct modeling without imputation. Columns were renamed to cleaner, standardized identifiers for easier handling during preprocessing. The target variable originally encoded as text (“Dropout”, “Enrolled”, “Graduate”) was label-encoded into numeric categories (0, 1, 2) to support model training. Numerical features such as admission grade, previous qualification grade, semester grades, and economic indicators were standardized using StandardScaler, while categorical features were retained as-is because the final models used tree-based algorithms capable of handling integer-encoded categorical inputs. The dataset was stratified and split into 80% training and 20% testing, preserving class distribution across the three student outcomes.

**Modeling Approach**

The modeling strategy focused on building a multi-class classification system to predict student outcomes: Dropout, Enrolled, or Graduate. Two major ensemble learning methods (Random Forest and XGBoost) were selected owing to their robustness, ability to model nonlinear relationships, and suitability for high-dimensional datasets. Both models were integrated into pipelines consisting of preprocessing and supervised learning components.
To improve generalization, RandomizedSearchCV with 3-fold cross-validation was applied to tune key hyperparameters such as tree depth, number of estimators, learning rate (for XGBoost), and sampling parameters. Accuracy was used as the scoring metric during optimization. After hyperparameter tuning, the best-performing configuration was refit on the full training set. Confusion matrices for both training and testing datasets were generated to evaluate class-wise performance and detect misclassification patterns. Finally, feature importance from the optimized XGBoost model was extracted to understand which academic, demographic, and socioeconomic variables contributed most to the predictions.

**Results**

The Random Forest model achieved a training accuracy of 81.58% and a test accuracy of 75.14%, indicating moderate generalization with some misclassifications across the three outcome categories. The XGBoost model outperformed Random Forest, reaching 83.50% on the training set and 76.95% on the test set, demonstrating stronger predictive capability and better handling of the underlying structure in the data.
Confusion matrix visualizations revealed that both models were particularly effective at identifying graduates, followed by enrolled students, while dropout predictions were the most challenging due to overlapping patterns in academic performance and demographic factors. Feature importance analysis highlighted key predictors such as semester grade averages, admission grade, unemployment rate, inflation rate, GDP variation, and curricular unit performance, suggesting that both academic consistency and external economic environment influence dropout likelihood. A sample inference using the XGBoost model demonstrated its practical use by estimating a student’s dropout probability at 0.94 for a randomly selected case.


<img width="902" height="363" alt="image" src="https://github.com/user-attachments/assets/6221e305-4da7-4cfd-9e7d-3db1a4836cdd" />

                                                  Figure 8

<img width="902" height="322" alt="image" src="https://github.com/user-attachments/assets/647fc081-3890-468e-93a5-450a9a3d7eb2" />

                                                  Figure 9

<img width="902" height="445" alt="image" src="https://github.com/user-attachments/assets/dc8d7d77-1c41-49aa-8343-d4b33216eb80" />

                                                  Figure 10

**Discussion**

The results suggest that academic performance indicators, especially first- and second-semester grades, play a critical role in determining student outcomes; students with consistently low grades show a markedly higher dropout risk. Socioeconomic factors such as unemployment and inflation rates also contributed significantly, indicating that external pressures may influence students’ ability to persist. Additionally, students with lower parental education and lower-skilled parental occupations were found to drop out more often, supporting the literature around social support and academic persistence.
The drop in accuracy from training to testing across both models indicates moderate overfitting, but still acceptable for predictive modeling in educational settings. The superior performance of XGBoost highlights the benefit of boosting algorithms, especially in handling imbalanced and complex multi-class problems. Nevertheless, dropout prediction remains challenging due to overlapping characteristics between enrolled and dropout groups, implying that integrating longitudinal behavioral data (attendance, engagement metrics, financial stress) could further improve accuracy.

**Conclusion**

This project successfully demonstrated a complete pipeline for predicting student dropout risk using demographic, academic, and socioeconomic features. The data preparation process ensured clean, structured inputs, and the modeling approach—centered on ensemble learning and hyperparameter tuning—yielded strong predictive performance. XGBoost emerged as the best model, providing reliable accuracy and interpretable feature importance to guide institutional decision-making.
The analysis highlights that academic achievement, parental background, and broader economic conditions are strong determinants of dropout risk. These insights can support early-warning systems that identify vulnerable students, enabling institutions to deploy targeted interventions such as tutoring programs, counseling, financial assistance, or academic support workshops.

                                               
