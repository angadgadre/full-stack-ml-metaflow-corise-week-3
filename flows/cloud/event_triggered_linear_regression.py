from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, catch, retry, timeout
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

@trigger(events=['s3'])
@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2'})
class TaxiFarePrediction(FlowSpec):

    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):

        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import MinMaxScaler

        obviously_bad_data_filters = [

        df.fare_amount > 0,         # fare_amount in US Dollars
        df.trip_distance <= 100,    # trip_distance in miles
        df.trip_distance > 0,

        # done: add some logic to filter out what you decide is bad data!
        # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.
        df.passenger_count > 0,
        # missing values removal
        ~df.trip_distance.isnull(),
        ~df.passenger_count.isnull(),
        ~df.fare_amount.isnull(),
        ~df.total_amount.isnull(),
        ~df.tip_amount.isnull(),
        ~df.congestion_surcharge.isnull(),
        ~df.tolls_amount.isnull(),
        ~df.payment_type.isnull(),
        ~df.mta_tax.isnull(),
        ~df.hour.isnull()
        ]

        for f in obviously_bad_data_filters:
            _df = df[f]

        # done: 
            # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
            # Understand what is happening.
            # Revisit task 1 and think about what might go in this function.
        
        # shortlist the features worth engineering
        _df = df.filter(items=['trip_distance', 'hour', 'passenger_count', 'fare_amount', 'total_amount', 'tip_amount', 'congestion_surcharge', 'airport_fee', 'tolls_amount', 'payment_type', 'mta_tax'])
        # Numerical features: 
        _ = _df.select_dtypes(include=['int64', 'float64']).columns
        
        
        # categorical features: airport_fee, payment_type, mta_tax
        _df['airport_fee'] = _df['airport_fee'].astype(str)
        _df['payment_type'] = _df['payment_type'].astype(str)
        _df['mta_tax'] = _df['mta_tax'].astype(str)

        # boolean features: none

        # datetime features: not doing at the moment    
        return _df

    @catch(var="read_failure")
    @retry(times=2)
    @timeout(seconds=59)
    @step
    def start(self):

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from sklearn.metrics import roc_auc_score

        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import MinMaxScaler


        self.df = self.transform_features(pd.read_parquet(self.data_url))

        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
        # This is a simple/naive way to do this, and is meant to keep this example simple, to focus learning on deploying Metaflow flows.
        # In practice, you want split time series data in more sophisticated ways and run backtests. 
        
        SEED=89

        def train_validation_test_split(
            X, y, train_ratio: float, validation_ratio: float, test_ratio: float
        ):
            # Split up dataset into train and test, of which we split up the test.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1 - train_ratio), random_state=SEED
            )

            # Split up test into two (validation and test).
            X_val, X_test, y_val, y_test = train_test_split(
                X_test,
                y_test,
                test_size=(test_ratio / (test_ratio + validation_ratio)),
                random_state=SEED,
            )

            # Return the splits
            return X_train, X_val, X_test, y_train, y_val, y_test

        X, y = (
            self.df.filter(items=['hour', 'passenger_count', 'fare_amount', 'total_amount', 'tip_amount', 'congestion_surcharge', 'airport_fee', 'tolls_amount', 'payment_type', 'mta_tax']),
            self.df['trip_distance']
        )

        # Splits according to ratio of 80/10/10
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = train_validation_test_split(
            X, y, 0.8, 0.10, 0.10
        )

        # Select the numerical columns
        self.numerical_cols_X = self.X_train.select_dtypes(include=["int64", "float64"]).columns

        # Numerical pipeline
        self.num_pipeline = Pipeline([
            ('scaler', MinMaxScaler())
        ])

        # Select the categorical columns
        self.categorical_cols_X = self.X_train.select_dtypes(include=["object"]).columns

        # Numerical pipeline
        self.cat_pipeline = Pipeline([
            ('encoding', OneHotEncoder())
        ])

        self.preprocessor = ColumnTransformer([
            ('cat', self.cat_pipeline, self.categorical_cols_X),
            ('num', self.num_pipeline, self.numerical_cols_X)
        ])

        self.next(self.linear_model)

    @step
    def linear_model(self):
        "Fitting a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy='most_frequent')
        # done: Play around with the model if you are feeling it.
        # self.model = LinearRegression()
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('imputer', imputer),
            ('model', HistGradientBoostingRegressor())    
        ])

        self.next(self.validate)

    # @step ? branch flow
    def other_model(self):
        "Fitting multi variable, linear model to the data."
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', LinearRegression())    
        ])

        # pipeline.fit(X_train, y_train)
        # y_predict = pipeline.predict(X_val)
        # r2_score(y_predict, y_val).round(4)


    def gather_sibling_flow_run_results(self):

        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow 
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅" 
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [Markdown(icon), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(score), Markdown(msg)]
                rows.append(row)
            else:
                rows.append([Markdown("✅"), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(str(self.scores.mean())), Markdown("This run...")])
        return rows
                
    
    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.pipeline, self.X_test, self.y_test, cv=5)

        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(Table(self.gather_sibling_flow_run_results(), headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"]))
        self.next(self.end)

    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
