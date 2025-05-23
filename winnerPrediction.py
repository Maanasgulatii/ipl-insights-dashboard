import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from datasetPreprocessing import new_matchesDF
from scrollToTop import create_scroll_to_top_button


def app():
    # Header matching other pages
    st.markdown(
        '''
            <h1 style='text-align:center; color: #FF6347;'><strong> 🎲 PREDICTING WIN PROBABILITY FOR A TEAM 🎲 </strong></h1>
            <hr style="border-top: 3px solid #FF6347;">
        ''',
        unsafe_allow_html=True
    )

    # Team selection
    Teams = new_matchesDF.team1.unique().tolist()
    c1, c2 = st.columns(2)

    with c1:
        t1 = st.selectbox("Select Team 1", Teams)
    with c2:
        t2 = st.selectbox("Select Team 2", Teams)

    # Venue selection
    Cities = new_matchesDF['city'].dropna().unique().tolist()
    venue = st.selectbox("Select Venue", Cities)

    # Toss win assumption
    toss_options = [f"{t1} Wins Toss", f"{t2} Wins Toss", "Unknown (50% Chance)"]
    toss_winner = st.radio("Toss Winner", toss_options)

    if t1 == t2:
        st.markdown(
            f"<h5 style='text-align: center; color: red;'> ⚠ Oops! Looks Like Team1 and Team2 Are Same ⚠</h5>",
            unsafe_allow_html=True
        )
    else:
        Analyze = st.button('Analyze')

        if Analyze:
            st.markdown(
                f"<h3 style='text-align: center;'> {t1} vs {t2} </h3>",
                unsafe_allow_html=True
            )

            # Filter matches between the two teams
            t1_batting = new_matchesDF[
                (new_matchesDF['team1'] == t1) & (new_matchesDF['team2'] == t2)
            ]
            t2_batting = new_matchesDF[
                (new_matchesDF['team1'] == t2) & (new_matchesDF['team2'] == t1)
            ]
            total = pd.concat([t1_batting, t2_batting], ignore_index=True)

            if total.empty:
                st.markdown(
                    f"<h5 style='text-align: center; color: red;'> ⚠ {t1} and {t2} have not played any matches together ⚠</h5>",
                    unsafe_allow_html=True
                )
            else:
                # Prepare features for logistic regression
                # Feature 1: Win outcome (1 if t1 wins, 0 if t2 wins)
                total['t1_wins'] = (total['winner'] == t1).astype(int)
                # Feature 2: Toss win (1 if t1 wins toss, 0 otherwise)
                total['t1_toss_win'] = (total['toss_winner'] == t1).astype(int)
                # Feature 3: Home advantage (simplified: 1 if t1's city, 0 otherwise)
                # Define home cities (simplified mapping based on team names)
                home_cities = {
                    'Mumbai Indians': 'Mumbai',
                    'Chennai Super Kings': 'Chennai',
                    'Royal Challengers Bangalore': 'Bangalore',
                    'Royal Challengers Bengaluru': 'Bengaluru',
                    'Sunrisers Hyderabad': 'Hyderabad',
                    'Delhi Capitals': 'Delhi',
                    'Delhi Daredevils': 'Delhi',
                    'Kolkata Knight Riders': 'Kolkata',
                    'Kings XI Punjab': 'Mohali',
                    'Punjab Kings': 'Mohali',
                    'Rajasthan Royals': 'Jaipur',
                    'Gujarat Titans': 'Ahmedabad',
                    'Lucknow Super Giants': 'Lucknow'
                }
                t1_home_city = home_cities.get(t1, '')
                total['t1_home_advantage'] = (total['city'] == t1_home_city).astype(int)

                # Features and target
                X = total[['t1_toss_win', 't1_home_advantage']]
                y = total['t1_wins']

                # Train a logistic regression model
                model = LogisticRegression()
                model.fit(X, y)

                # Set toss win based on user selection
                if toss_winner == f"{t1} Wins Toss":
                    current_toss_win = 1
                elif toss_winner == f"{t2} Wins Toss":
                    current_toss_win = 0
                else:
                    current_toss_win = 0.5  # Unknown

                # Set home advantage based on selected venue
                current_home_advantage = 1 if t1_home_city and venue == t1_home_city else 0

                # Predict probability for the current match
                current_features = [[current_toss_win, current_home_advantage]]
                win_prob_t1 = model.predict_proba(current_features)[0][1] * 100  # Probability of t1 winning
                win_prob_t2 = 100 - win_prob_t1

                # Display win probabilities for both teams
                st.markdown(
                    f"<h4 style='text-align: center; color: white;'> Win Probability: {t1}: {win_prob_t1:.2f}% | {t2}: {win_prob_t2:.2f}% </h4>",
                    unsafe_allow_html=True
                )

                # Display model accuracy
                accuracy = model.score(X, y) * 100
                st.markdown(
                    f"<h5 style='text-align: center; color: white;'> Model Accuracy on Historical Data: {accuracy:.2f}% </h5>",
                    unsafe_allow_html=True
                )

                # Bar chart of win probabilities
                prob_df = pd.DataFrame({
                    'Team': [t1, t2],
                    'Win Probability (%)': [win_prob_t1, win_prob_t2]
                })
                fig = px.bar(
                    data_frame=prob_df,
                    x='Team',
                    y='Win Probability (%)',
                    title=f'Win Probability: {t1} vs {t2}',
                    color='Team',
                    color_discrete_map={t1: 'blue', t2: 'orange'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

                st.image("Images/divider.png")

                # Historical stats table
                total_matches = len(total)
                t1_wins = len(total[total['winner'] == t1])
                t2_wins = len(total[total['winner'] == t2])
                t1_toss_wins = len(total[total['toss_winner'] == t1])
                t2_toss_wins = len(total[total['toss_winner'] == t2])

                stats_df = pd.DataFrame({
                    'Metric': ['Total Matches', 'Wins', 'Toss Wins'],
                    t1: [total_matches, t1_wins, t1_toss_wins],
                    t2: [total_matches, t2_wins, t2_toss_wins]
                })

                st.markdown(
                    f"<h4 style='text-align: center; color: white;'> Historical Stats: {t1} vs {t2} </h4>",
                    unsafe_allow_html=True
                )
                st.table(stats_df)

                st.image("Images/divider.png")

                # Feature Importance Breakdown
                st.markdown(
                    f"<h4 style='text-align: center; color: white;'> Factors Influencing Prediction </h4>",
                    unsafe_allow_html=True
                )
                feature_names = ['Toss Win', 'Home Advantage']
                coefficients = model.coef_[0]
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': coefficients
                })
                fig = px.bar(
                    data_frame=feature_importance_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance in Prediction',
                    color='Feature',
                    color_discrete_map={'Toss Win': 'green', 'Home Advantage': 'purple'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

                st.image("Images/divider.png")

                # Recent Form Analysis
                st.markdown(
                    f"<h4 style='text-align: center; color: white;'> Recent Form (Last 5 Matches) </h4>",
                    unsafe_allow_html=True
                )
                # Get last 5 matches for each team
                t1_matches = new_matchesDF[
                    (new_matchesDF['team1'] == t1) | (new_matchesDF['team2'] == t1)
                ].sort_values(by='date', ascending=False).head(5)
                t2_matches = new_matchesDF[
                    (new_matchesDF['team1'] == t2) | (new_matchesDF['team2'] == t2)
                ].sort_values(by='date', ascending=False).head(5)

                # Calculate win rates
                t1_wins_recent = len(t1_matches[t1_matches['winner'] == t1])
                t2_wins_recent = len(t2_matches[t2_matches['winner'] == t2])
                t1_win_rate = (t1_wins_recent / 5) * 100 if len(t1_matches) == 5 else 0
                t2_win_rate = (t2_wins_recent / 5) * 100 if len(t2_matches) == 5 else 0

                # Display recent form in a table
                recent_form_df = pd.DataFrame({
                    'Metric': ['Matches Played', 'Wins', 'Win Rate (%)'],
                    t1: [len(t1_matches), t1_wins_recent, t1_win_rate],
                    t2: [len(t2_matches), t2_wins_recent, t2_win_rate]
                })
                st.table(recent_form_df)

                st.image("Images/divider.png")

    # Footer matching other pages
    create_scroll_to_top_button(key_suffix="winnerPrediction")
    st.image("Images/divider.png")