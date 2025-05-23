import pandas as pd
import streamlit as st

def app():
    st.markdown(
        '''<h1 style='text-align:center; color: #ffcd19;'><strong>💠 SCORE PREDICTION 💠</strong></h1>
            <hr style="border-top: 3px solid #ffcd19;">
        ''',
        unsafe_allow_html=True
    )

    TEAMS = [
        'Chennai Super Kings',
        'Delhi Capitals',
        'Kings XI Punjab',
        'Kolkata Knight Riders',
        'Mumbai Indians',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Batting Team At The Moment', TEAMS)
    with col2:
        bowling_team = st.selectbox('Bowling Team At The Moment', TEAMS)

    if bowling_team == batting_team:
        st.error("Bowling and Batting Team Can't Be The Same")
    else:
        encoded_batting_team = [
            1 if batting_team == TEAM else 0 for TEAM in TEAMS
        ]
        encoded_bowling_team = [
            1 if bowling_team == TEAM else 0 for TEAM in TEAMS
        ]

        current_runs = st.number_input(
            'Enter Current Score of Batting Team..',
            min_value=0,
            step=1
        )

        wickets_left = st.number_input(
            'Enter Number of Wickets Left For Batting Team..',
            min_value=0,
            step=1
        )

        wickets_out = int(10 - wickets_left)

        over = st.number_input(
            'Current Over of The Match..',
            min_value=0,
            step=1
        )

        run_lst_5 = st.number_input(
            'How Many Runs Batting Team Has Scored In Last 5 Overs ?',
            min_value=0,
            step=1
        )

        wicket_lst_5 = st.number_input(
            'Number of  Wickets Taken By Bowling Team In The Last 5 Overs ?',
            min_value=0,
            step=1
        )

        data = [
            int(current_runs),
            int(wickets_out),
            over,
            int(run_lst_5),
            int(wicket_lst_5)
        ]

        data.extend(encoded_batting_team)
        data.extend(encoded_bowling_team)

        st.write('---')

        st.write('Encoded Input Data:', pd.DataFrame([data]))

        Generate_pred = st.button("Predict Score")

        if Generate_pred:
            overs_spent = over
            remaining_overs = 20 - overs_spent

            current_rr = current_runs / (overs_spent + 0.1)
            last_5_rr = run_lst_5 / 5
            base_rr = (current_rr * 0.4 + last_5_rr * 0.6)

            if wickets_out <= 2:
                adjusted_rr = base_rr * 1.05
            elif wickets_out <= 5:
                adjusted_rr = base_rr * 0.95
            elif wickets_out <= 8:
                adjusted_rr = base_rr * 0.85
            else:
                adjusted_rr = base_rr * 0.75

            projected_score = current_runs + (adjusted_rr * remaining_overs)
            lower = int(projected_score - 5)
            upper = int(projected_score + 5)

            st.subheader(
                f'📊 The Predicted Score Will Be Between {lower} - {upper}'
            )
