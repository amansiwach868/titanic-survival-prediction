# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Adjusting some settings for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Loading the training dataset
train_dff = pd.read_csv('kaggle_datasets/train.csv')
train_df = train_dff.copy()

# Displaying the first 5 rows to get an overview of the dataset
train_df.head()

# Overview of the dataset's main features
train_df.info()

# Examining missing values in the dataset
train_df.isnull().sum()

# Visualizing missing values using a heatmap
sns.heatmap(train_df.isnull(), cmap='viridis', annot=False)
plt.title('# Missing data analysis for the training dataset (train_df)')
plt.show()

# Descriptive statistics for numerical variables in train_df
print(train_df.describe().T)

# There were 2 missing values in the categorical variable 'Embarked', so we filled them with the mode.
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# We will impute the missing values in the 'Age' variable using a more accurate groupby approach instead of a simple statistic like mean or median.
train_df['Age'] = train_df['Age'].fillna(train_df.groupby(['Embarked', 'Sex'])['Age'].transform('mean'))

# Calculating the percentage of missing values in the 'Cabin' column relative to the total number of observations
round(100 * train_df['Cabin'].isnull().sum() / len(train_df), 3)  # 77.104

# Since 77.1% of the values in the 'Cabin' column are missing, instead of imputing them, we will convert this feature into a new variable indicating whether a cabin value is present or not.
train_df['CABIN_BOOL'] = train_df['Cabin'].notnull().astype(int)

# The 'Cabin' feature is no longer needed, so we will drop it from the dataset.
train_df = train_df.drop('Cabin', axis=1)


# After handling missing data, we will now proceed with outlier detection and analysis.

# The 'outlier_threshold' function calculates lower and upper bounds for outlier detection using the IQR method and visualizes them with histplots.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75, hist=False):
    """
    Calculates the lower and upper bounds to detect outliers in a numerical variable
    using the Interquartile Range (IQR) method.

    Parameters:
    ----------
    :param dataframe: pandas.DataFrame
        The dataset containing the variable to analyze.
    :param col_name: str
        The column name (numeric) for which outliers will be detected.
    :param q1: float, optional (default=0.25)
        The lower quantile for IQR calculation.
    :param q3: float, optional (default=0.75)
        The upper quantile for IQR calculation.
    :param hist: bool, optional (default=False)
    If True, displays a boxplot with the calculated outlier thresholds for visual analysis.
    :return:
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    if hist:
        sns.boxplot(x=dataframe[col_name], color='skyblue')
        plt.axvline(low_limit, color='red', linestyle='--', label='Lower Bound')
        plt.axvline(up_limit, color='red', linestyle='--', label='Upper Bound')
        plt.title(f"Boxplot of {col_name} with Outlier Thresholds")
        plt.xlabel(f"{col_name}")
        plt.legend()
        plt.show()
    return low_limit, up_limit


outlier_thresholds(train_df, 'Age', hist=True)

# As these operations are specific to numerical variables,
# we'll define a function to classify variables by their data types
# and store them in lists for later use.

# Since 'PassengerID' does not carry any meaningful information, we exclude it from the list of numerical variables (num_cols).
train_df = train_df.drop('PassengerId', axis=1)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and categorical but cardinal variables in the dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataset containing the variable to analyze.
    cat_th : int, float
        Class threshold for numerical but categorical variables.
    car_th : int, float
        Class threshold for categorical but cardinal variables.

    Returns
    -------
    cat_cols : list
        List of categorical variables
    num_cols : list
        List of numerical variables
    cat_but_car : list
        List of categorical but cardinal variables

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is included in cat_cols
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


# We have categorized the variables based on their data types and stored their names in separate lists.
cat_cols, num_cols, cat_but_car = grab_col_names(train_df)


# With the check_outlier function, we can easily check whether a variable contains outliers.
# To preserve potential insights arising from multivariate interactions, I set q1 and q3 to 0.05 and 0.95 in this analysis.

def check_outlier(dataframe, col_name, q1=0.25, q3=0.95):
    """

    :param dataframe: pandas.DataFrame
        The dataset containing the variable to analyze.
    :param col_name: str
        The column name (numeric) for which outliers will be detected.
    :param q1: float, optional (default=0.25)
        The lower quantile for IQR calculation.
    :param q3: float, optional (default=0.75)
        The upper quantile for IQR calculation.
    :return: bool
    Returns True if outliers are detected in the specified variable, otherwise False.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(f"{col}: {'Contains outliers' if check_outlier(train_df, col, q1=0.05, q3=0.95) else 'No outliers detected'}")


# With the replace_with_threshold function, we will cap the outliers using the threshold values obtained from the outlier_thresholds function.
def replace_with_threshold(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_threshold(train_df, 'Fare', q1=0.05, q3=0.95)

# At this point, we have resolved both missing value and outlier issues. Now, we can proceed to feature generation.

train_df['TICKET_LENGTH'] = train_df['Ticket'].apply(lambda x: len(str(x)))
train_df = train_df.drop('Ticket', axis=1)

train_df["NEW_TITLE"] = train_df.Name.str.extract(' ([A-Za-z]+).', expand=False)
train_df = train_df.drop('Name', axis=1)

train_df['AGE_GROUPED'] = pd.cut(train_df['Age'], bins=(0, 25, 50, train_df['Age'].max()),
                                 labels=['young', 'adult', 'senior'])

train_df['FARE_CAT'] = pd.cut(train_df['Fare'], bins=[0, 50, 100, train_df['Fare'].max()],
                              labels=['cheap', 'expensive', 'luxury'])

train_df.head()

# We are now moving on to the encoding phase.
# First, we will handle binary categorical variables using LabelEncoding.

binary_cols = [col for col in train_df.columns if train_df[col].nunique() == 2 and col not in ['Survived', 'CABIN_BOOL']]

le = LabelEncoder()

for col in binary_cols:
    train_df[col] = le.fit_transform(train_df[col])

train_df['Sex'].value_counts()


# Before proceeding with one-hot encoding, we will apply rare encoding to handle infrequent categories.

def rare_encoder(dataframe, rare_perc):
    """
    Combines infrequent categories in categorical variables under a single 'Rare' label
    based on a given frequency threshold.
    :param dataframe: pandas.DataFrame
        The input dataframe containing categorical variables.
    :param rare_perc: float
        The frequency threshold. Categories with a relative frequency below this value will be grouped as 'Rare'.
    :return: pandas.DataFrame
        A dataframe where rare categories in object-type columns are replaced with the label 'Rare'.

        Notes
    -----
    - Only object-type (categorical) columns are considered.
    - This transformation helps reduce dimensionality before encoding.
    """
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns
                    if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


train_df = rare_encoder(train_df, 0.01)

# We have completed the binary and rare encoding step.
# Now, we are moving on to the one-hot encoding phase.

cat_cols, num_cols, cat_but_car = grab_col_names(train_df)

cat_cols = [col for col in cat_cols if col not in ['Survived', 'CABIN_BOOL', 'Sex'] ]

cat_cols = cat_cols + cat_but_car

# Before applying one-hot encoding, we will use the grab_col_names function to update our variable lists
# and remove any unnecessary features from cat_cols.

train_df = pd.get_dummies(train_df, columns = cat_cols, drop_first=True, dtype=int)

# And finally, we apply RobustScaler to our numerical features.

rs = RobustScaler()

train_df[num_cols] = rs.fit_transform(train_df[num_cols])

# Our dataset is now ready for machine learning algorithms.
# Let's train a logistic regression model and evaluate its performance metrics together.

y = train_df['Survived']

X = train_df.drop('Survived', axis = 1)

# Split the dataset into training and test sets (using stratify to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1 = report['weighted avg']['f1-score']
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ROC-Curve
RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Generate confusion matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()