#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind, ttest_1samp, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

"""# Features:
- enrollee_id : Unique ID for candidate
- city: City code
- city_ development _index : Developement index of the city (scaled)
- gender: Gender of candidate
- relevent_experience: Relevant experience of candidate
- enrolled_university: Type of University course enrolled if any
- education_level: Education level of candidate
- major_discipline :Education major discipline of candidate
- experience: Candidate total experience in years
- company_size: No of employees in current employer's company
- company_type : Type of current employer
- lastnewjob: Difference in years between previous job and current job
- training_hours: training hours completed
- target: 0 – Not looking for job change, 1 – Looking for a job change

#Inspecting data
"""

jobs_df = pd.read_csv('data/aug_train.csv')
pd.set_option('display.max_columns', None)

"""#Make some columns be categorical"""

jobs_df['target'] = pd.Categorical(jobs_df['target'], [0, 1], ordered=True)
jobs_df['education_level'].fillna('No education', inplace=True)
jobs_df['education_level'] = pd.Categorical(jobs_df['education_level'], ['No education', 'Primary School', 'High School', 'Graduate', 'Masters', 'Phd'], ordered=True)
jobs_df['relevent_experience'] = pd.Categorical(jobs_df['relevent_experience'], ['No relevent experience', 'Has relevent experience'], ordered=True)
jobs_df['experience'].fillna('No experience', inplace=True)
jobs_df['experience'] = pd.Categorical(jobs_df['experience'],
                                       ['No experience', '<1', '1', '2', '3' , '4', '5', '6', '7', '8', '9', '10',
                                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'], ordered=True)

"""#Data description"""

print(jobs_df.head())
print(jobs_df.info())
print(len(jobs_df)) ## rows
jobs_desc = jobs_df.describe()
print(jobs_desc)

"""#Comparing educational level to relevent experience, education level, major discipline, company_size, company_type, target"""

comparing_ed_level_with =["relevent_experience", "education_level","major_discipline", "company_size","company_type", "target"]
for i, comparison in enumerate(comparing_ed_level_with):
    plt.figure(figsize=[10, 10])
    sns.countplot(x=comparison, hue='education_level', edgecolor="black", alpha=0.7, data=jobs_df)
    sns.despine()
    plt.title("Countplot of {}  by education_level".format(comparison))
    plt.show()

"""#Compare experience to education level"""

plt.figure(figsize=[15,4])
sns.countplot(x='experience', hue='education_level',edgecolor="black", alpha=0.7, data=jobs_df)
sns.despine()
plt.title("Countplot of experience by education_level")
plt.show()

"""#Compare training hours to education level"""

plt.figure(figsize=(7, 7))
sns.barplot(data=jobs_df, x='education_level', y='training_hours')
plt.title('Countplot of training hours by education level')
plt.show()

"""#Make copy of dataframe to calculate mean education level and relevent experience depending on education level"""

jobs_cp = jobs_df.copy()

#Get Unique tables
unique_rel_exp = jobs_cp['relevent_experience'].unique().sort_values()
unique_exp = jobs_cp['experience'].unique().sort_values()
print(unique_exp, unique_rel_exp)
#Make this categorical table numeric
jobs_cp['relevent_experience'] = jobs_cp['relevent_experience'].cat.codes
jobs_cp['experience'] = jobs_cp['experience'].cat.codes

"""#Calculate mean"""

education_levels = ['High School', 'Graduate', 'Masters', 'Phd']
columns = ['experience', 'relevent_experience', 'training_hours']
for education_level in education_levels:
    print(f'VALUES FOR {education_level}')
    for column in columns:
        df = jobs_cp[jobs_cp['education_level'] == education_level]
        mean = round(np.mean(df[column]))
        median = round(np.median(df[column]))
        if column == 'experience':
            print(f'Mean value of {column} is {unique_exp[mean]} years')
            print(f'Median value of {column} is {unique_exp[median]} years')
        if column == 'relevent_experience':
            print(f'Mean value of {column} is {unique_rel_exp[mean]}')
            print(f'Median value of {column} is {unique_rel_exp[median]}')
        if column == 'training_hours':
            print(f'Mean value of {column} is {mean} hours')
            print(f'Median value of {column} is {median} hours')
    print('\n')

"""#Check association between education level and target"""

xtab = pd.crosstab(jobs_df['education_level'], jobs_df['target'])
chi2, pval, dof, exp = chi2_contingency(xtab)
print('''
H0 - there is a dependence between education level and the desire to find a job in the contingency table.
HA - there is no dependence between education level and the desire to find a job in the contingency table.
''')
result = 'Reject H0' if pval < 0.05 else "Accept H0"
print(pval)
print(result)

"""#Check association between education level and city"""

xtab = pd.crosstab(jobs_df['education_level'], jobs_df['city'])
chi2, pval, dof, exp = chi2_contingency(xtab)
print('''
H0 - there is a dependence between education level and city in the contingency table.
HA - there is no dependence between education level and city in the contingency table.
''')
result = 'Reject H0' if pval < 0.05 else "Accept H0"
print(pval)
print(result)

"""#Check association between desure to find a job and city"""

xtab = pd.crosstab(jobs_df['target'], jobs_df['city'])
chi2, pval, dof, exp = chi2_contingency(xtab)
print('''
H0 - there is a dependence between desire to find a job and city in the contingency table.
HA - there is no dependence between desire to find a job and city in the contingency table.
''')
result = 'Reject H0' if pval < 0.05 else "Accept H0"
print(pval)
print(result)

"""#Check association between gender and target"""

xtab = pd.crosstab(jobs_df['gender'], jobs_df['target'])
chi2, pval, dof, exp = chi2_contingency(xtab)
print('''
H0 - there is a dependence between gender and the desire to find a job in the contingency table.
HA - there is no dependence between gender and the desire to find a job in the contingency table.
''')
result = 'Reject H0' if pval < 0.05 else "Accept H0"
print(pval)
print(result)

"""#Testing association between gender and training hours"""

man_training = jobs_df[jobs_df['gender'] == 'Male']['training_hours']
woman_training = jobs_df[jobs_df['gender'] == 'Female']['training_hours']
tstat, pval = ttest_ind(man_training, woman_training)
print('''
H0 - there is a dependence between gender and training hours.
HA - there is no dependence between gender and training hours.
''')
result = 'Reject H0' if pval < 0.05 else "Accept H0"
print(pval)
print(result)

"""#Testing association between education level and training hours"""

grad_training = jobs_df[jobs_df['education_level'] == 'Graduate']['training_hours']
masters_training = jobs_df[jobs_df['education_level'] == 'Masters']['training_hours']
phd_training = jobs_df[jobs_df['education_level'] == 'Phd']['training_hours']
high_school_training = jobs_df[jobs_df['education_level'] == 'High School']['training_hours']
tstat, pval = f_oneway(grad_training, masters_training, phd_training, high_school_training)
print('''
H0 - there is a dependence between educational level and training hours.
HA - there is no dependence between educational level and training hours.
''')
result = 'Reject H0' if pval < 0.05 else "Accept H0"
print(pval)
print(result)

"""#There is a dependence between education level and training hours so we need to check where"""

print(jobs_df['education_level'].unique())
tukey_results = pairwise_tukeyhsd(jobs_df['training_hours'].astype('float'), jobs_df['education_level'], 0.05)
print(tukey_results)

"""#We have high dependence between education level and training hours"""
