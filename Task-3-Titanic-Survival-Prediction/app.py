import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from model_utils import load_model, load_raw_dataset, predict_passenger


st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon=":material/directions_boat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(
    style="whitegrid",
    rc={"axes.spines.top": False, "axes.spines.right": False},
)


@st.cache_resource
def get_model_artifact():
    return load_model()


@st.cache_data
def get_dataset():
    return load_raw_dataset()


def render_page_heading(title, description):
    st.header(title)
    st.caption(description)


def render_confusion_matrix(model_name, values):
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.heatmap(
        values,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="#FFFFFF",
        xticklabels=["Did not survive", "Survived"],
        yticklabels=["Did not survive", "Survived"],
        annot_kws={"size": 12, "weight": "bold"},
        ax=axis,
    )
    axis.set_title(f"{model_name} · validation results", loc="left", pad=14)
    axis.set_xlabel("Predicted outcome", labelpad=10)
    axis.set_ylabel("Actual outcome", labelpad=10)
    st.pyplot(figure)
    plt.close(figure)


st.title("Titanic Survival Prediction", icon=":material/directions_boat:")
st.badge("Machine learning application", icon=":material/analytics:", color="blue")
st.caption("A supervised machine-learning study of passenger survival outcomes.")

try:
    artifact = get_model_artifact()
except FileNotFoundError as error:
    st.error(str(error))
    st.code("python train.py")
    st.stop()
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()

with st.sidebar:
    st.subheader("Workspace")
    page = st.radio(
        "Navigate to a section",
        ["Survival prediction", "Model performance", "Dataset insights"],
        label_visibility="visible",
    )
    st.space("small")
    st.caption("SELECTED MODEL")
    st.markdown(f"**{artifact['model_name']}**")
    st.caption(
        f"Validation ROC-AUC · {artifact['validation_metrics']['roc_auc']:.1%}"
    )


if page == "Survival prediction":
    render_page_heading(
        "Predict Titanic Survival",
        "Enter passenger details to estimate survival using a trained machine-learning model.",
    )

    form_column, result_column = st.columns([1.1, 0.9], gap="large")

    with form_column.container(border=True):
        st.subheader("Passenger information")
        with st.form("passenger_details", border=False):
            st.caption("PASSENGER DETAILS")
            class_column, gender_column = st.columns(2)
            with class_column:
                pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
            with gender_column:
                sex = st.selectbox("Gender", ["female", "male"])
            age = st.number_input(
                "Age in years",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                help="Choose an age between 0 and 100 years.",
            )

            st.caption("FAMILY INFORMATION")
            siblings_column, parents_column = st.columns(2)
            with siblings_column:
                sibsp = st.number_input(
                    "Siblings and spouses",
                    min_value=0,
                    max_value=10,
                    value=0,
                    help="Number of siblings or spouses traveling with the passenger.",
                )
            with parents_column:
                parch = st.number_input(
                    "Parents and children",
                    min_value=0,
                    max_value=10,
                    value=0,
                    help="Number of parents or children traveling with the passenger.",
                )

            st.caption("TRAVEL INFORMATION")
            fare_column, port_column = st.columns(2)
            with fare_column:
                fare = st.number_input(
                    "Ticket fare",
                    min_value=0.0,
                    max_value=1000.0,
                    value=32.0,
                    help="Fare paid for the ticket, in historical dataset units.",
                )
            with port_column:
                embarked = st.selectbox(
                    "Port of embarkation",
                    ["S", "C", "Q"],
                    format_func=lambda value: {
                        "S": "Southampton",
                        "C": "Cherbourg",
                        "Q": "Queenstown",
                    }[value],
                )

            submitted = st.form_submit_button(
                "Predict Survival",
                type="primary",
                width="stretch",
                icon=":material/query_stats:",
            )

    if submitted:
        passenger = {
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
        }
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_error", None)
        try:
            with st.spinner("Calculating the prediction..."):
                prediction, probability = predict_passenger(
                    artifact["model"], passenger
                )
            st.session_state["prediction_result"] = {
                "passenger": passenger,
                "prediction": prediction,
                "probability": probability,
            }
        except (TypeError, ValueError, KeyError) as error:
            st.session_state["prediction_error"] = str(error)
        except Exception as error:
            st.session_state["prediction_error"] = (
                f"The prediction could not be completed: {error}"
            )

    with result_column.container(border=True):
        st.subheader("Prediction result")
        if error_message := st.session_state.get("prediction_error"):
            st.error(error_message)
        elif result := st.session_state.get("prediction_result"):
            prediction = result["prediction"]
            probability = result["probability"]
            if prediction == 1:
                st.success(
                    "PASSENGER PREDICTED TO SURVIVE",
                    icon=":material/check_circle:",
                )
            else:
                st.warning(
                    "PASSENGER PREDICTED NOT TO SURVIVE",
                    icon=":material/error:",
                )

            if probability is not None:
                survival_column, non_survival_column = st.columns(2)
                survival_column.metric("Survival probability", f"{probability:.1%}")
                non_survival_column.metric(
                    "Non-survival probability", f"{1 - probability:.1%}"
                )
                st.progress(probability)
                st.caption(
                    "Probabilities are the model's estimates for this passenger; "
                    "they are not guarantees."
                )
            else:
                st.caption("This model does not provide probability estimates.")

            passenger = result["passenger"]
            passenger_summary = pd.DataFrame(
                {
                    "Passenger detail": [
                        "Passenger class",
                        "Gender",
                        "Age",
                        "Siblings and spouses",
                        "Parents and children",
                        "Ticket fare",
                        "Port of embarkation",
                    ],
                    "Entered value": [
                        f"Class {passenger['pclass']}",
                        passenger["sex"].title(),
                        f"{passenger['age']:g} years",
                        str(passenger["sibsp"]),
                        str(passenger["parch"]),
                        f"{passenger['fare']:.2f}",
                        {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[
                            passenger["embarked"]
                        ],
                    ],
                }
            )
            st.dataframe(
                passenger_summary,
                width="stretch",
                hide_index=True,
                alt="Passenger details used for this survival prediction",
            )
        else:
            st.info(
                "Your prediction will appear here. Complete the passenger details "
                "and select Predict Survival.",
                icon=":material/info:",
            )

elif page == "Model performance":
    render_page_heading(
        "Model Performance",
        "Compare the classifiers using the same held-out validation passengers.",
    )

    selected_name = artifact["model_name"]
    selected_metrics = artifact["validation_metrics"]
    with st.container(border=True):
        st.subheader("Selected model")
        selected_columns = st.columns([1.2, 1, 1.8], vertical_alignment="center")
        selected_columns[0].metric("Model", selected_name)
        selected_columns[1].metric(
            "Validation ROC-AUC", f"{selected_metrics['roc_auc']:.1%}"
        )
        selected_columns[2].write(
            "Selection rule: highest validation ROC-AUC; F1-score breaks ties."
        )

    st.info(
        f"{selected_name} was selected by validation ROC-AUC, not by accuracy. "
        "These validation scores were used for model selection and are not an "
        "independent final test estimate.",
        icon=":material/info:",
    )

    st.subheader("Classifier comparison")
    result_rows = artifact["model_comparison"]
    comparison_table = pd.DataFrame(
        [
            {
                "Model": result["model_name"],
                "Selection": "Selected" if result["model_name"] == selected_name else "",
                "ROC-AUC": f"{result['roc_auc']:.1%}",
                "Accuracy": f"{result['accuracy']:.1%}",
                "Precision": f"{result['precision']:.1%}",
                "Recall": f"{result['recall']:.1%}",
                "F1-score": f"{result['f1']:.1%}",
            }
            for result in result_rows
        ]
    )
    st.dataframe(
        comparison_table,
        width="stretch",
        hide_index=True,
        alt="Validation metric comparison for the three classifiers",
    )

    chart_data = pd.DataFrame(result_rows).set_index("model_name")
    st.bar_chart(
        chart_data[["accuracy", "precision", "recall", "f1", "roc_auc"]],
        y_label="Validation score",
        height=320,
        alt="Validation scores for accuracy, precision, recall, F1-score, and ROC-AUC",
    )

    st.subheader("Confusion matrices")
    st.caption(
        "Each matrix counts predictions on the same 179 validation passengers."
    )
    matrix_tabs = st.tabs(list(artifact["validation_confusion_matrices"]))
    for (model_name, matrix), tab in zip(
        artifact["validation_confusion_matrices"].items(), matrix_tabs
    ):
        with tab:
            render_confusion_matrix(model_name, matrix)

    with st.expander("How to read these evaluation metrics"):
        st.markdown(
            "- **Accuracy:** share of all passengers classified correctly.\n"
            "- **Precision:** among passengers predicted to survive, the share who survived.\n"
            "- **Recall:** among passengers who survived, the share the model identified.\n"
            "- **F1-score:** a balance of precision and recall.\n"
            "- **ROC-AUC:** how well the model ranks survivors above non-survivors across thresholds.\n"
            "- **Confusion matrix:** counts correct and incorrect predictions by actual and predicted outcome."
        )

elif page == "Dataset insights":
    render_page_heading(
        "Dataset Insights",
        "Explore passenger demographics, ticket classes, and survival patterns in the Titanic dataset.",
    )

    try:
        dataset = get_dataset()
    except (OSError, ValueError) as error:
        st.error(f"Could not load the Titanic dataset: {error}")
        st.stop()

    passenger_count = len(dataset)
    survivor_count = int(dataset["survived"].sum())
    non_survivor_count = passenger_count - survivor_count
    survival_rate = survivor_count / passenger_count if passenger_count else 0.0
    with st.container(horizontal=True):
        st.metric("Total passengers", f"{passenger_count:,}", border=True)
        st.metric("Survivors", f"{survivor_count:,}", border=True)
        st.metric("Did not survive", f"{non_survivor_count:,}", border=True)
        st.metric("Overall survival rate", f"{survival_rate:.1%}", border=True)

    plot_data = dataset.copy()
    plot_data["outcome"] = plot_data["survived"].map(
        {0: "Did not survive", 1: "Survived"}
    )
    outcome_order = ["Did not survive", "Survived"]
    outcome_colors = {"Did not survive": "#B91C1C", "Survived": "#15803D"}

    first_column, second_column = st.columns(2, gap="large")
    with first_column:
        st.subheader("Survival distribution")
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.countplot(
            data=plot_data,
            x="outcome",
            hue="outcome",
            order=outcome_order,
            hue_order=outcome_order,
            palette=outcome_colors,
            legend=False,
            ax=axis,
        )
        axis.set_xlabel("Passenger outcome")
        axis.set_ylabel("Passengers")
        for container in axis.containers:
            axis.bar_label(container, fmt="%d", padding=3)
        st.pyplot(figure)
        plt.close(figure)

        st.subheader("Survival by passenger class")
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.countplot(
            data=plot_data,
            x="pclass",
            hue="outcome",
            hue_order=outcome_order,
            palette=outcome_colors,
            ax=axis,
        )
        axis.set_xlabel("Passenger class")
        axis.set_ylabel("Passengers")
        axis.legend(title="Outcome", loc="upper right")
        st.pyplot(figure)
        plt.close(figure)

        st.subheader("Ticket fare distribution")
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.histplot(
            data=plot_data,
            x="fare",
            hue="outcome",
            hue_order=outcome_order,
            bins=28,
            element="step",
            palette=outcome_colors,
            ax=axis,
        )
        axis.set_xlabel("Ticket fare")
        axis.set_ylabel("Passengers")
        axis.legend(title="Outcome", labels=outcome_order)
        st.pyplot(figure)
        plt.close(figure)

    with second_column:
        st.subheader("Survival by gender")
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.countplot(
            data=plot_data,
            x="sex",
            hue="outcome",
            hue_order=outcome_order,
            palette=outcome_colors,
            ax=axis,
        )
        axis.set_xlabel("Gender")
        axis.set_ylabel("Passengers")
        axis.legend(title="Outcome", loc="upper right")
        st.pyplot(figure)
        plt.close(figure)

        st.subheader("Age distribution")
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.histplot(
            data=plot_data,
            x="age",
            hue="outcome",
            hue_order=outcome_order,
            bins=24,
            element="step",
            palette=outcome_colors,
            ax=axis,
        )
        axis.set_xlabel("Age in years")
        axis.set_ylabel("Passengers")
        axis.legend(title="Outcome", labels=outcome_order)
        st.pyplot(figure)
        plt.close(figure)
        st.caption("Passengers with missing ages are omitted from this chart.")

        st.subheader("Missing values")
        missing_counts = dataset.isna().sum().loc[lambda values: values > 0]
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        sns.barplot(
            x=missing_counts.index,
            y=missing_counts.values,
            color="#2563EB",
            ax=axis,
        )
        axis.set_xlabel("Dataset field")
        axis.set_ylabel("Missing passengers")
        axis.tick_params(axis="x", rotation=20)
        for container in axis.containers:
            axis.bar_label(container, fmt="%d", padding=3)
        st.pyplot(figure)
        plt.close(figure)

    with st.expander("Preview the source dataset"):
        st.dataframe(
            dataset.head(10),
            width="stretch",
            hide_index=True,
            alt="First ten rows of the Titanic passenger dataset",
        )