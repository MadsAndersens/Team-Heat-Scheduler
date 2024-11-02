import streamlit as st
import pandas as pd
import numpy as np
import random

import pulp
import itertools

def generate_schedule(num_flights, num_boats, num_teams, num_heats):
    # Create the problem
    prob = pulp.LpProblem("Race_Scheduling", pulp.LpMinimize)
    
    # Indices
    teams = list(range(num_teams))
    flights = list(range(num_flights))
    heats = list(range(num_heats))
    boats = list(range(num_boats))
    team_pairs = list(itertools.combinations(teams, 2))
    
    # Decision variables
    x = pulp.LpVariable.dicts("x", (teams, flights, heats, boats), cat='Binary')
    w = pulp.LpVariable.dicts("w", (team_pairs, flights, heats), cat='Binary')
    
    # Random weights for objective function
    random_weights = {}
    for t in teams:
        for f in flights:
            for h in heats:
                for b in boats:
                    random_weights[t, f, h, b] = random.random()
    
    # Penalty weights for team pairs not racing together
    penalty_w = {}
    for (t1, t2) in team_pairs:
        penalty_w[(t1, t2)] = random.random()
    
    # Objective function: minimize total random weights and penalties
    prob += (
        pulp.lpSum([random_weights[t, f, h, b] * x[t][f][h][b] 
                    for t in teams for f in flights for h in heats for b in boats]) +
        pulp.lpSum([penalty_w[(t1, t2)] * (1 - pulp.lpSum([w[(t1, t2)][f][h] for f in flights for h in heats]))
                    for (t1, t2) in team_pairs])
    ), "Objective"
    
    # Constraints
    
    # 1. Each boat in each flight and heat is assigned to at most one team
    for f in flights:
        for h in heats:
            for b in boats:
                prob += pulp.lpSum([x[t][f][h][b] for t in teams]) <= 1, f"Boat_{b}_Flight_{f}_Heat_{h}"
    
    # 2. Each team in each flight and heat is assigned to at most one boat
    for t in teams:
        for f in flights:
            for h in heats:
                prob += pulp.lpSum([x[t][f][h][b] for b in boats]) <= 1, f"Team_{t}_Flight_{f}_Heat_{h}"
    
    
    # 3. Define w variables to indicate if two teams are racing together
    for (t1, t2) in team_pairs:
        for f in flights:
            for h in heats:
                prob += w[(t1, t2)][f][h] <= pulp.lpSum([x[t1][f][h][b] for b in boats]), f"W1_{t1}_{t2}_Flight_{f}_Heat_{h}"
                prob += w[(t1, t2)][f][h] <= pulp.lpSum([x[t2][f][h][b] for b in boats]), f"W2_{t1}_{t2}_Flight_{f}_Heat_{h}"
                prob += w[(t1, t2)][f][h] >= pulp.lpSum([x[t1][f][h][b] for b in boats]) + \
                                                      pulp.lpSum([x[t2][f][h][b] for b in boats]) - 1, f"W3_{t1}_{t2}_Flight_{f}_Heat_{h}"
    
    # 4. Every team must sail at least once per flight
    for t in teams:
        for f in flights:
            prob += pulp.lpSum([x[t][f][h][b] for h in heats for b in boats]) == 1, f"Team_{t}_Flight_{f}_Participation"
    

    # Solve the problem with a time limit of 300 seconds (5 minutes)
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=st.session_state['time_limit'])
    result = prob.solve(solver)

    # Check if a feasible solution was found
    if pulp.LpStatus[result] in ['Infeasible', 'Unbounded', 'Undefined']:
        st.warning("No feasible solution found.")
        return None
    
    # Extract the schedule
    schedule = []
    for f in flights:
        flight_schedule = []
        for h in heats:
            heat_schedule = {}
            for b in boats:
                for t in teams:
                    if pulp.value(x[t][f][h][b]) == 1:
                        heat_schedule[b] = t + 1  # Adjusting team index to start from 1
            flight_schedule.append(heat_schedule)
        schedule.append(flight_schedule)
    
    return schedule

def create_html_table(schedule, num_boats, num_heats):
    html_table = "<table border='1' cellpadding='10'>"
    
    # Create the flights and heats structure
    for flight_index, flight in enumerate(schedule):
        html_table += f"<tr><th colspan='{num_boats + 1}'>Flight {flight_index + 1}</th></tr>"
        for heat_index, heat in enumerate(flight):
            race_number = flight_index * len(flight) + heat_index + 1
            html_table += f"<tr><td>Heat {heat_index + 1}</td>"
            for boat in range(num_boats):
                html_table += f"<td>Boat {boat + 1}</td>"
            html_table += "</tr><tr><td></td>"
            for boat in range(num_boats):
                team = heat.get(boat, "")
                html_table += f"<td>{team}</td>"
            html_table += "</tr>"
    html_table += "</table>"
    return html_table

# Streamlit UI
st.title("Team Heat Scheduler")

#Columns with equal width
col1, col2 = st.columns(2)

# Input fields for the user
with col1:
    num_flights = st.number_input("Number of Flights", min_value=1, max_value=20, value=5)
    num_boats = st.number_input("Number of Boats", min_value=2, max_value=10, value=6)

with col2: 
    num_teams = st.number_input("Number of Teams", min_value=2, max_value=20, value=12)
    num_heats = st.number_input("Number of Heats per Flight", min_value=1, max_value=6, value=2)

# Time limit for the solver
st.subheader("How long can the program think?")
st.session_state['time_limit'] = st.number_input("Solver Time Limit (seconds)",
                                                 min_value=1,
                                                 max_value=600,
                                                  value=300)

# Generate schedule button
if st.button("Generate Schedule"):
    with st.spinner("Generating Schedule based on input..."):
        schedule = generate_schedule(num_flights, num_boats, num_teams, num_heats)
    
    if schedule is not None:
        st.write("### Generated Schedule")
        html_table = create_html_table(schedule, num_boats, num_heats)
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.write("No feasible schedule could be generated with the given parameters.")
