"""Complete implementation of 13 interactive football visualizations."""

import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import os
import logging

# --- Configuration ---
DB_PATH = "C:/Users/shumayil/Downloads/Visualization_Project_2/Visualization-Project-football/database.sqlite"
OUTPUT_DIR = "C:/Users/shumayil/Downloads/Visualization_Project_2/Visualization-Project-football/visualizations"
MIN_MINUTES_PLAYED_CONSISTENCY = 900  # Example threshold for consistency analysis

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. Database Connection and Data Loading ---
def load_data(db_path=DB_PATH):
    """Loads necessary tables from the SQLite database."""
    logging.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        logging.info("Loading tables...")
        players = pd.read_sql_query("SELECT * FROM Player", conn)
        player_attributes = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
        teams = pd.read_sql_query("SELECT * FROM Team", conn)
        team_attributes = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
        matches = pd.read_sql_query("SELECT * FROM Match", conn)
        leagues = pd.read_sql_query("SELECT * FROM League", conn)
        countries = pd.read_sql_query("SELECT * FROM Country", conn)
        logging.info("Tables loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")
    return players, player_attributes, teams, team_attributes, matches, leagues, countries


# --- 2. Data Preprocessing Utilities ---
def calculate_age(birth_date_str, on_date_str="2016-01-01"):
    """Calculates age given a birth date string and a reference date string."""
    if pd.isna(birth_date_str):
        return np.nan
    try:
        birth_date = datetime.strptime(birth_date_str.split(" ")[0], "%Y-%m-%d")
        on_date = datetime.strptime(on_date_str, "%Y-%m-%d")
        age = on_date.year - birth_date.year - ((on_date.month, on_date.day) < (birth_date.month, birth_date.day))
        return age
    except (ValueError, TypeError):
        return np.nan

def preprocess_data(players, player_attributes, teams, team_attributes, matches, leagues, countries):
    """Performs necessary preprocessing and merging."""
    logging.info("Starting data preprocessing...")

    # Player data
    players['age'] = players['birthday'].apply(calculate_age)
    player_attributes['date'] = pd.to_datetime(player_attributes['date'])
    latest_player_attributes = player_attributes.loc[player_attributes.groupby("player_api_id")["date"].idxmax()]
    players_merged = pd.merge(players, latest_player_attributes, on="player_api_id", how="inner")
    logging.info(f"Players merged shape: {players_merged.shape}")

    # Match data
    matches['date'] = pd.to_datetime(matches['date'])
    matches['goal_difference'] = matches['home_team_goal'] - matches['away_team_goal']
    matches['match_outcome'] = np.select([
        matches['goal_difference'] > 0,
        matches['goal_difference'] < 0
    ], [
        "Home Win", "Away Win"
    ], default="Draw")
    matches = pd.merge(matches, teams[["team_api_id", "team_long_name"]].rename(columns={"team_long_name": "home_team_name"}), left_on="home_team_api_id", right_on="team_api_id", how="left")
    matches = pd.merge(matches, teams[["team_api_id", "team_long_name"]].rename(columns={"team_long_name": "away_team_name"}), left_on="away_team_api_id", right_on="team_api_id", how="left")
    matches = pd.merge(matches, leagues[["id", "name"]].rename(columns={"name": "league_name"}), left_on="league_id", right_on="id", how="left")
    logging.info(f"Matches merged shape: {matches.shape}")

    # Team attributes (get latest)
    team_attributes['date'] = pd.to_datetime(team_attributes['date'])
    latest_team_attributes = team_attributes.loc[team_attributes.groupby("team_api_id")["date"].idxmax()]
    teams_merged = pd.merge(teams, latest_team_attributes, on="team_api_id", how="inner")
    logging.info(f"Teams merged shape: {teams_merged.shape}")

    # Add league name to players (requires match data - complex, skipping for now)
    # Add team name to players
    # This requires linking player_api_id to team_api_id via matches, which is tricky as players move teams.
    # For simplicity, we might use the team associated with the latest attribute record if needed, but it''s not robust.

    logging.info("Data preprocessing finished.")
    return players_merged, player_attributes, teams_merged, team_attributes, matches

# --- 3. Visualization Functions ---

# VISUALIZATION 1: Interactive Player Potential Matrix
def create_player_potential_matrix(df):
    logging.info("Creating Visualization 1: Player Potential Matrix")
    df_filtered = df[(df["potential"].notna()) & (df["overall_rating"].notna()) & (df["age"].notna()) &
                     (df["potential"] > 60) & (df["overall_rating"] > 60) & (df["age"] >= 17) & (df["age"] <= 38)].copy()
    # Simulate market value for size encoding (as it''s not in the dataset)
    df_filtered.loc[:, "market_value_sim"] = ((df_filtered["potential"] + df_filtered["overall_rating"]) / 2 + (df_filtered["potential"] - df_filtered["overall_rating"])) * np.random.uniform(0.1, 1.5, len(df_filtered))
    df_filtered.loc[:, "market_value_sim"] = df_filtered["market_value_sim"].clip(lower=1).round(1)

    fig = px.scatter(df_filtered, x="overall_rating", y="potential", size="market_value_sim", color="age",
                     hover_name="player_name",
                     hover_data={"age": True, "player_api_id": False, "market_value_sim": ":.1fM"},
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title="Interactive Player Potential Matrix")
    fig.add_shape(type="line", x0=df_filtered["overall_rating"].min(), y0=df_filtered["overall_rating"].min(),
                  x1=max(df_filtered["overall_rating"].max(), df_filtered["potential"].max()),
                  y1=max(df_filtered["overall_rating"].max(), df_filtered["potential"].max()),
                  line=dict(dash="dash", color="grey"))
    fig.update_layout(xaxis_title="Current Overall Rating", yaxis_title="Potential Rating",
                      coloraxis_colorbar_title_text="Age", xaxis_range=[60,100], yaxis_range=[60,100])
    # Add annotations (adjust positions based on data range)
    fig.add_annotation(x=70, y=95, text="High Potential Gems", showarrow=True, arrowhead=1, ax=-40, ay=-40)
    fig.add_annotation(x=95, y=95, text="Established Stars", showarrow=True, arrowhead=1, ax=40, ay=-30)
    return fig

# VISUALIZATION 2: Dynamic Player Attributes Comparison (Radar Chart)
def create_player_attributes_radar(df, player_names_list):
    logging.info("Creating Visualization 2: Player Attributes Radar")
    default_radar_attributes = [
        "attacking_work_rate", "defensive_work_rate", # Convert work rate to numeric?
        "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
        "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control",
        "acceleration", "sprint_speed", "agility", "reactions", "balance",
        "shot_power", "jumping", "stamina", "strength", "long_shots",
        "aggression", "interceptions", "positioning", "vision", "penalties",
        "marking", "standing_tackle", "sliding_tackle",
        "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"
    ]
    # Select a subset of key attributes for default view
    key_attrs = ["sprint_speed", "dribbling", "finishing", "short_passing", "standing_tackle", "strength", "vision", "reactions"]
    data_to_plot = df[df["player_name"].isin(player_names_list)].copy()
    if data_to_plot.empty:
        logging.warning("No data found for selected players in radar chart.")
        return go.Figure(layout_title_text="No data for selected players.")

    # Handle potential non-numeric work rates (simple mapping)
    work_rate_map = {"low": 1, "medium": 2, "high": 3}
    if "attacking_work_rate" in data_to_plot.columns:
        data_to_plot['attacking_work_rate_num'] = data_to_plot['attacking_work_rate'].map(work_rate_map).fillna(0) * 33
    if "defensive_work_rate" in data_to_plot.columns:
        data_to_plot['defensive_work_rate_num'] = data_to_plot['defensive_work_rate'].map(work_rate_map).fillna(0) * 33
    # Update key_attrs if work rate was included
    # key_attrs = [attr for attr in key_attrs if attr in data_to_plot.columns]

    fig = go.Figure()
    for _, row in data_to_plot.iterrows():
        values = row[key_attrs].fillna(0).tolist()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=key_attrs,
            fill="toself",
            name=row["player_name"],
            hoverinfo="r+theta+name"
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Player Attributes Comparison"
    )
    return fig

# VISUALIZATION 3: Advanced Team Performance & Improvement Analyzer
def create_team_performance_analyzer(matches_df_local, teams_df_local, league_name="England Premier League"):
    logging.info("Creating Visualization 3: Team Performance Analyzer")
    league_matches = matches_df_local[matches_df_local["league_name"] == league_name].copy()
    if league_matches.empty:
        logging.warning(f"No match data found for league: {league_name}")
        return go.Figure(layout_title_text=f"No data for {league_name}")

    def get_points(row):
        if row["home_team_goal"] > row["away_team_goal"]: return 3, 0
        if row["home_team_goal"] < row["away_team_goal"]: return 0, 3
        return 1, 1

    points = league_matches.apply(get_points, axis=1, result_type='expand')
    league_matches.loc[:, "home_team_points"] = points[0]
    league_matches.loc[:, "away_team_points"] = points[1]

    home_season_stats = league_matches.groupby(["season", "home_team_api_id"])["home_team_points"].sum().reset_index()
    away_season_stats = league_matches.groupby(["season", "away_team_api_id"])["away_team_points"].sum().reset_index()
    home_season_stats.rename(columns={"home_team_api_id": "team_api_id", "home_team_points": "points"}, inplace=True)
    away_season_stats.rename(columns={"away_team_api_id": "team_api_id", "away_team_points": "points"}, inplace=True)

    season_stats = pd.concat([home_season_stats, away_season_stats])
    total_season_stats = season_stats.groupby(["season", "team_api_id"])["points"].sum().reset_index()
    total_season_stats = pd.merge(total_season_stats, teams_df_local[["team_api_id", "team_long_name"]], on="team_api_id")

    # Calculate rank per season
    total_season_stats['rank'] = total_season_stats.groupby("season")["points"].rank(method="dense", ascending=False)

    # Select top teams based on average rank or points
    avg_points = total_season_stats.groupby("team_long_name")["points"].mean().sort_values(ascending=False)
    top_teams = avg_points.head(8).index.tolist() # Show more teams
    plot_df = total_season_stats[total_season_stats["team_long_name"].isin(top_teams)]

    fig = px.line(plot_df, x="season", y="points", color="team_long_name",
                  title=f"Team Performance Trajectory (Points) in {league_name}",
                  markers=True,
                  hover_name="team_long_name",
                  hover_data={"season": True, "points": True, "rank": True})
    fig.update_layout(xaxis_title="Season", yaxis_title="Total Points", legend_title="Team")
    return fig

# VISUALIZATION 4: Interactive Possession & Tactical Outcome Explorer
def create_possession_outcome_explorer(df):
    logging.info("Creating Visualization 4: Possession Outcome Explorer")
    # NOTE: Possession data is NOT in the provided Match table.
    # We MUST simulate it for demonstration purposes.
    logging.warning("Possession data not available. Simulating for demonstration.")
    df_sample = df.sample(n=min(2000, len(df)), random_state=42).copy() # Sample for performance
    df_sample.loc[:, "possession_home_sim"] = np.random.normal(50, 15, len(df_sample)).clip(10, 90)
    df_sample.loc[:, "possession_away_sim"] = 100 - df_sample["possession_home_sim"]

    fig = px.scatter(
        df_sample,
        x="possession_home_sim",
        y="goal_difference",
        color="match_outcome",
        hover_name="home_team_name",
        hover_data={"away_team_name": True, "home_team_goal": True, "away_team_goal": True, "league_name": True, "season": True},
        title="Possession vs. Match Outcome Explorer (Simulated Possession)",
        color_discrete_map={"Home Win": "green", "Away Win": "red", "Draw": "blue"},
        opacity=0.7
    )
    fig.add_vline(x=50, line_dash="dash", line_color="grey")
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(xaxis_title="Home Team Possession % (Simulated)", yaxis_title="Goal Difference (Home - Away)")
    # Quadrant annotations
    fig.add_annotation(x=75, y=df_sample["goal_difference"].max()*0.8, text="High Poss, Winning", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
    fig.add_annotation(x=25, y=df_sample["goal_difference"].min()*0.8, text="Low Poss, Losing", showarrow=False, bgcolor="rgba(255,0,0,0.1)")
    fig.add_annotation(x=25, y=df_sample["goal_difference"].max()*0.8, text="Low Poss, Winning (Counter?)", showarrow=False, bgcolor="rgba(0,0,255,0.1)")
    fig.add_annotation(x=75, y=df_sample["goal_difference"].min()*0.8, text="High Poss, Losing (Inefficient?)", showarrow=False, bgcolor="rgba(255,165,0,0.1)")
    return fig

# VISUALIZATION 5: Player Development Trajectories by Position
def create_player_development_trajectories(player_attr_df, players_df_local):
    logging.info("Creating Visualization 5: Player Development Trajectories")
    # Use the full player_attributes table for time series data
    df_dev = pd.merge(player_attr_df, players_df_local[["player_api_id", "birthday"]], on="player_api_id")
    df_dev['age'] = df_dev.apply(lambda row: calculate_age(row['birthday'], row['date'].strftime('%Y-%m-%d')), axis=1)
    df_dev = df_dev.dropna(subset=['age'])
    df_dev = df_dev[df_dev['age'] <= 40]

    # Define age groups
    age_bins = [16, 21, 24, 27, 30, 33, 41]
    age_labels = ["<21", "21-23", "24-26", "27-29", "30-32", ">32"]
    df_dev['age_group'] = pd.cut(df_dev["age"], bins=age_bins, labels=age_labels, right=False)

    # Simulate primary position (this is a major simplification)
    # A better approach would involve analyzing typical attributes for positions
    logging.warning("Simulating player positions based on attributes for Vis 5.")
    def assign_position(row):
        if row['gk_diving'] > 50: return "Goalkeeper"
        if row['standing_tackle'] > 70 or row['sliding_tackle'] > 70: return "Defender"
        if row['dribbling'] > 70 or row['short_passing'] > 70: return "Midfielder"
        if row['finishing'] > 70 or row['shot_power'] > 70: return "Forward"
        return "Midfielder" # Default
    df_dev['primary_position'] = df_dev.apply(assign_position, axis=1)

    # Attributes for physical and technical/mental
    physical_attrs = ["acceleration", "sprint_speed", "stamina", "strength", "agility", "balance"]
    technical_mental_attrs = ["dribbling", "finishing", "short_passing", "ball_control", "vision", "reactions", "positioning"]

    df_dev_melted = df_dev.melt(id_vars=["age_group", "primary_position"], value_vars=physical_attrs + technical_mental_attrs, var_name="attribute", value_name="rating")
    df_dev_melted.dropna(subset=["rating", "age_group"], inplace=True)
    avg_ratings = df_dev_melted.groupby(["age_group", "primary_position", "attribute"])["rating"].mean().reset_index()

    # Create interactive plot with dropdown for position
    fig = go.Figure()
    positions = df_dev['primary_position'].unique()
    buttons = []

    default_position = "Midfielder"

    for i, position in enumerate(positions):
        visible = (position == default_position)
        # Physical attributes traces
        plot_data_phys = avg_ratings[(avg_ratings["attribute"].isin(physical_attrs)) & (avg_ratings["primary_position"] == position)]
        for attribute in physical_attrs:
            attr_data = plot_data_phys[plot_data_phys["attribute"] == attribute]
            fig.add_trace(go.Scatter(x=attr_data["age_group"], y=attr_data["rating"], name=f"{attribute} (Phys)",
                                     legendgroup="Physical", legendgrouptitle_text="Physical", visible=visible,
                                     mode='lines+markers', hovertemplate=f"<b>{attribute}</b><br>Age: %{{x}}<br>Avg Rating: %{{y:.1f}}<extra></extra>"))

        # Technical/Mental attributes traces
        plot_data_tech = avg_ratings[(avg_ratings["attribute"].isin(technical_mental_attrs)) & (avg_ratings["primary_position"] == position)]
        for attribute in technical_mental_attrs:
            attr_data = plot_data_tech[plot_data_tech["attribute"] == attribute]
            fig.add_trace(go.Scatter(x=attr_data["age_group"], y=attr_data["rating"], name=f"{attribute} (Tech/Mental)",
                                     legendgroup="Technical/Mental", legendgrouptitle_text="Technical/Mental", visible=visible,
                                     mode='lines+markers', hovertemplate=f"<b>{attribute}</b><br>Age: %{{x}}<br>Avg Rating: %{{y:.1f}}<extra></extra>"))

        # Create button logic
        visibility_mask = [(trace.legendgroup == "Physical" or trace.legendgroup == "Technical/Mental") and (position in trace.name if hasattr(trace, 'name') else False) for trace in fig.data]
        # Correction: visibility needs to be based on position, not trace name content
        num_phys_attrs = len(physical_attrs)
        num_tech_attrs = len(technical_mental_attrs)
        num_attrs_per_pos = num_phys_attrs + num_tech_attrs
        visibility_mask = [False] * len(fig.data)
        start_index = i * num_attrs_per_pos
        end_index = start_index + num_attrs_per_pos
        for j in range(start_index, end_index):
            if j < len(visibility_mask):
                 visibility_mask[j] = True

        buttons.append(dict(label=position,
                            method="update",
                            args=[{"visible": visibility_mask},
                                  {"title": f"Player Development Trajectories for {position}s"}]))

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[dict(active=positions.tolist().index(default_position),
                          buttons=buttons,
                          direction="down",
                          pad={"r": 10, "t": 10},
                          showactive=True,
                          x=0.1, xanchor="left",
                          y=1.1, yanchor="top")],
        title_text=f"Player Development Trajectories for {default_position}s",
        xaxis_title="Age Group",
        yaxis_title="Average Rating",
        height=600,
        legend_tracegroup_general_attrs_visible=False # Hide subplot titles in legend
    )
    # Set initial visibility correctly
    initial_visibility = [False] * len(fig.data)
    default_pos_index = positions.tolist().index(default_position)
    start_index = default_pos_index * num_attrs_per_pos
    end_index = start_index + num_attrs_per_pos
    for j in range(start_index, end_index):
        if j < len(initial_visibility):
            initial_visibility[j] = True
    for k, trace in enumerate(fig.data):
        trace.visible = initial_visibility[k]

    return fig

# VISUALIZATION 6: Player Role Fingerprint (Simulated Petal Plot using Radar)
def create_player_role_fingerprint(df, player_name):
    logging.info("Creating Visualization 6: Player Role Fingerprint")
    player_data = df[df["player_name"] == player_name].iloc[0] if not df[df["player_name"] == player_name].empty else None
    if player_data is None:
        logging.warning(f"Player {player_name} not found for fingerprint.")
        return go.Figure(layout_title_text=f"Player {player_name} not found.")

    # Define skill categories and relevant attributes
    skill_categories = {
        "Pace & Movement": ["acceleration", "sprint_speed", "agility"],
        "Shooting": ["finishing", "long_shots", "shot_power", "volleys"],
        "Passing": ["short_passing", "long_passing", "vision", "crossing"],
        "Dribbling & Control": ["dribbling", "ball_control", "balance"],
        "Defending": ["standing_tackle", "sliding_tackle", "interceptions", "marking"],
        "Physicality": ["strength", "stamina", "jumping", "aggression"],
        "Goalkeeping": ["gk_diving", "gk_handling", "gk_kicking", "gk_reflexes"]
    }

    # Calculate average score for each category
    category_scores = {}
    for category, attrs in skill_categories.items():
        valid_attrs = [attr for attr in attrs if attr in player_data and pd.notna(player_data[attr])]
        if valid_attrs:
            category_scores[category] = player_data[valid_attrs].mean()
        else:
            category_scores[category] = 0

    # Filter out Goalkeeping for non-GK, or other categories for GK
    is_gk = category_scores.get("Goalkeeping", 0) > 40 # Simple heuristic
    if is_gk:
        final_categories = {k: v for k, v in category_scores.items() if k == "Goalkeeping" or "gk" in k.lower()}
    else:
        final_categories = {k: v for k, v in category_scores.items() if k != "Goalkeeping"}

    if not final_categories:
         return go.Figure(layout_title_text=f"No valid categories for {player_name}.")

    # Use Radar chart to simulate petal plot
    labels = list(final_categories.keys())
    values = list(final_categories.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=player_name,
        hovertemplate='<b>%{theta}</b><br>Avg Rating: %{r:.1f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title=f"Player Role Fingerprint: {player_name}",
        showlegend=False
    )
    return fig

# VISUALIZATION 7: Team Tactical Matchup Matrix (Interactive Heatmap)
def create_team_tactical_matchup_matrix(matches_df_local, teams_merged_df_local):
    logging.info("Creating Visualization 7: Team Tactical Matchup Matrix")
    # Define Team Archetypes based on latest team attributes (simplistic)
    logging.warning("Defining team archetypes based on limited attributes for Vis 7.")
    teams_attrs = teams_merged_df_local[["team_api_id", "team_long_name", "buildUpPlaySpeed", "buildUpPlayDribbling", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing", "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth"]].copy()
    teams_attrs.fillna(teams_attrs.mean(numeric_only=True), inplace=True)

    # Simple Archetype Logic (Example - needs refinement)
    def assign_archetype(row):
        if row["buildUpPlaySpeed"] > 70 and row["chanceCreationShooting"] > 60: return "Fast Attack / Direct"
        if row["buildUpPlayPassing"] < 40 and row["defenceAggression"] > 60: return "High Press / Counter"
        if row["buildUpPlayPassing"] > 60 and row["defencePressure"] < 40: return "Possession / Patient"
        if row["defenceAggression"] > 55 and row["defenceTeamWidth"] < 45: return "Defensive / Compact"
        return "Balanced"
    teams_attrs['archetype'] = teams_attrs.apply(assign_archetype, axis=1)
    archetype_map = teams_attrs.set_index('team_api_id')['archetype']

    # Calculate performance against archetypes
    results = []
    for team_id, team_data in matches_df_local.groupby("home_team_api_id"):
        team_name = team_data['home_team_name'].iloc[0]
        for opp_id, opp_matches in team_data.groupby("away_team_api_id"):
            opp_archetype = archetype_map.get(opp_id, "Unknown")
            if opp_archetype != "Unknown":
                wins = (opp_matches['goal_difference'] > 0).sum()
                draws = (opp_matches['goal_difference'] == 0).sum()
                losses = (opp_matches['goal_difference'] < 0).sum()
                matches_played = len(opp_matches)
                points = wins * 3 + draws * 1
                ppg = points / matches_played if matches_played > 0 else 0
                results.append({"team_id": team_id, "team_name": team_name, "opponent_archetype": opp_archetype, "ppg": ppg, "matches": matches_played})

    # Aggregate results
    matrix_data = pd.DataFrame(results)
    pivot_data = matrix_data.groupby(['team_name', 'opponent_archetype']).agg(avg_ppg=('ppg', 'mean'), total_matches=('matches', 'sum')).reset_index()
    pivot_table = pivot_data.pivot(index='team_name', columns='opponent_archetype', values='avg_ppg')

    # Limit to teams with enough matches for stability
    teams_with_enough_data = pivot_data.groupby('team_name')['total_matches'].sum()
    teams_to_show = teams_with_enough_data[teams_with_enough_data > 50].index # Example threshold
    pivot_table_filtered = pivot_table[pivot_table.index.isin(teams_to_show)].dropna(axis=1, how='all') # Drop archetypes with no data

    if pivot_table_filtered.empty:
        logging.warning("Not enough data to create matchup matrix.")
        return go.Figure(layout_title_text="Not enough data for Matchup Matrix.")

    fig = px.imshow(pivot_table_filtered, 
                    text_auto=".2f", 
                    aspect="auto",
                    color_continuous_scale='RdYlGn', 
                    title="Team Performance (Avg PPG) vs. Opponent Archetypes (Simulated)",
                    labels=dict(color="Avg PPG", x="Opponent Archetype", y="Team"))
    fig.update_xaxes(side="bottom")
    fig.update_layout(height=800)
    return fig

# VISUALIZATION 8: Key Player Contribution Network (Conceptual Network using Scatter)
def create_player_contribution_network(matches_df_local, players_df_local, team_name, season="2015/2016"):
    logging.info("Creating Visualization 8: Player Contribution Network")
    # This requires detailed player involvement data per match (passes, assists, goals) which is not readily available.
    # We will simulate a network based on co-occurrence in matches for a specific team and season.
    logging.warning("Simulating player network based on co-occurrence for Vis 8.")

    team_matches = matches_df_local[(matches_df_local['season'] == season) & 
                                    ((matches_df_local['home_team_name'] == team_name) | (matches_df_local['away_team_name'] == team_name))]
    if team_matches.empty:
        logging.warning(f"No matches found for {team_name} in {season}.")
        return go.Figure(layout_title_text=f"No matches for {team_name} in {season}.")

    # Extract player IDs involved in these matches (simplified - taking home/away players)
    player_cols = [f"{side}_player_{i}" for side in ["home", "away"] for i in range(1, 12)]
    involved_player_ids = pd.unique(team_matches[player_cols].values.ravel('K'))
    involved_player_ids = involved_player_ids[~np.isnan(involved_player_ids)] # Remove NaNs

    # Get player names
    player_info = players_df_local[players_df_local['player_api_id'].isin(involved_player_ids)][['player_api_id', 'player_name']].set_index('player_api_id')

    # Simulate connections (e.g., players appearing in the same match lineup)
    edges = {}
    for _, row in team_matches.iterrows():
        lineup_ids = [row[col] for col in player_cols if pd.notna(row[col])]
        for i in range(len(lineup_ids)):
            for j in range(i + 1, len(lineup_ids)):
                p1, p2 = sorted((lineup_ids[i], lineup_ids[j]))
                if p1 != p2:
                    edges[(p1, p2)] = edges.get((p1, p2), 0) + 1

    if not edges:
        logging.warning(f"No player connections found for {team_name} in {season}.")
        return go.Figure(layout_title_text=f"No connections for {team_name} in {season}.")

    # Prepare data for Plotly scatter plot (nodes and edges)
    nodes_x = np.random.rand(len(involved_player_ids)) * 100
    nodes_y = np.random.rand(len(involved_player_ids)) * 100
    node_ids = involved_player_ids.tolist()
    node_map = {pid: i for i, pid in enumerate(node_ids)}

    edge_x = []
    edge_y = []
    edge_weights = []
    for (p1, p2), weight in edges.items():
        if p1 in node_map and p2 in node_map:
            x0, y0 = nodes_x[node_map[p1]], nodes_y[node_map[p1]]
            x1, y1 = nodes_x[node_map[p2]], nodes_y[node_map[p2]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(weight) # Store weight if needed for line width/color

    # Create node trace
    node_trace = go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text',
                          marker=dict(size=15, color='lightblue'),
                          text=[player_info.loc[pid]['player_name'] if pid in player_info.index else f"ID: {int(pid)}" for pid in node_ids],
                          textposition='top center',
                          hoverinfo='text',
                          hovertext=[player_info.loc[pid]['player_name'] if pid in player_info.index else f"ID: {int(pid)}" for pid in node_ids])

    # Create edge trace
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                          line=dict(width=1, color='gray'),
                          hoverinfo='none')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Conceptual Player Network for {team_name} ({season}) - Based on Co-occurrence',
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        hovermode='closest'
                    ))
    return fig

# VISUALIZATION 9: League Competitiveness Dashboard
def create_league_competitiveness_dashboard(matches_df_local, teams_df_local, league_name="England Premier League"):
    logging.info("Creating Visualization 9: League Competitiveness Dashboard")
    league_matches = matches_df_local[matches_df_local['league_name'] == league_name].copy()
    if league_matches.empty:
        logging.warning(f"No match data found for league: {league_name}")
        return go.Figure(layout_title_text=f"No data for {league_name}")

    # Calculate points per team per season (reusing logic from Vis 3)
    def get_points(row):
        if row["home_team_goal"] > row["away_team_goal"]: return 3, 0
        if row["home_team_goal"] < row["away_team_goal"]: return 0, 3
        return 1, 1
    points = league_matches.apply(get_points, axis=1, result_type='expand')
    league_matches.loc[:, "home_team_points"] = points[0]
    league_matches.loc[:, "away_team_points"] = points[1]
    home_season_stats = league_matches.groupby(["season", "home_team_api_id"])["home_team_points"].sum().reset_index()
    away_season_stats = league_matches.groupby(["season", "away_team_api_id"])["away_team_points"].sum().reset_index()
    home_season_stats.rename(columns={"home_team_api_id": "team_api_id", "home_team_points": "points"}, inplace=True)
    away_season_stats.rename(columns={"away_team_api_id": "team_api_id", "away_team_points": "points"}, inplace=True)
    season_stats = pd.concat([home_season_stats, away_season_stats])
    total_season_stats = season_stats.groupby(["season", "team_api_id"])["points"].sum().reset_index()
    total_season_stats = pd.merge(total_season_stats, teams_df_local[["team_api_id", "team_long_name"]], on="team_api_id")

    # Calculate metrics per season
    competitiveness_metrics = []
    for season, data in total_season_stats.groupby("season"):
        sorted_data = data.sort_values("points", ascending=False).reset_index()
        if len(sorted_data) < 4: continue # Need at least 4 teams
        gap_1_2 = sorted_data.loc[0, "points"] - sorted_data.loc[1, "points"]
        gap_1_4 = sorted_data.loc[0, "points"] - sorted_data.loc[3, "points"]
        # Gini coefficient calculation
        points_array = np.sort(sorted_data["points"].values)
        n = len(points_array)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * points_array)) / (n * np.sum(points_array)) if np.sum(points_array) > 0 else 0
        competitiveness_metrics.append({
            "season": season,
            "gap_1st_2nd": gap_1_2,
            "gap_1st_4th": gap_1_4,
            "gini_coefficient": gini,
            "num_teams": n
        })
    metrics_df = pd.DataFrame(competitiveness_metrics)

    # Create dashboard with subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Points Gap (1st vs 2nd)", "Points Gap (1st vs 4th)", "League Point Inequality (Gini)", "Number of Teams"))

    fig.add_trace(go.Scatter(x=metrics_df["season"], y=metrics_df["gap_1st_2nd"], name="Gap 1st-2nd", mode='lines+markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=metrics_df["season"], y=metrics_df["gap_1st_4th"], name="Gap 1st-4th", mode='lines+markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=metrics_df["season"], y=metrics_df["gini_coefficient"], name="Gini Coeff", mode='lines+markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=metrics_df["season"], y=metrics_df["num_teams"], name="Num Teams", mode='lines+markers'), row=2, col=2)

    fig.update_layout(title=f"League Competitiveness Dashboard: {league_name}", height=700, showlegend=False)
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Points", row=1, col=1)
    fig.update_yaxes(title_text="Points", row=1, col=2)
    fig.update_yaxes(title_text="Gini Coefficient", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    return fig

# VISUALIZATION 10: Interactive Shot Map & Expected Goals (xG) Analyzer (Conceptual)
def create_shot_map_conceptual(matches_df_local, team_name="Arsenal", season="2015/2016"):
    logging.info("Creating Visualization 10: Conceptual Shot Map")
    # Highly conceptual - requires shot location (x,y) and xG data which is not present.
    logging.warning("Simulating shot data (locations, outcome, xG) for Vis 10.")

    team_matches = matches_df_local[((matches_df_local['home_team_name'] == team_name) | (matches_df_local['away_team_name'] == team_name)) & (matches_df_local['season'] == season)]
    if team_matches.empty:
        logging.warning(f"No matches found for {team_name} in {season} for shot map.")
        return go.Figure(layout_title_text=f"No matches for {team_name} in {season}.")

    # Simulate shots for these matches
    num_shots_to_simulate = 150
    sim_shots = pd.DataFrame({
        "match_id": np.random.choice(team_matches['match_api_id'].unique(), num_shots_to_simulate),
        "x": np.random.uniform(50, 100, num_shots_to_simulate), # Attacking half
        "y": np.random.uniform(5, 95, num_shots_to_simulate),  # Width of pitch
        "outcome": np.random.choice(["Goal", "Saved", "Miss/Blocked"], num_shots_to_simulate, p=[0.12, 0.35, 0.53]),
        "player": np.random.choice(["Player X", "Player Y", "Player Z", "Other"], num_shots_to_simulate, p=[0.3, 0.3, 0.2, 0.2]),
        "xg_simulated": np.random.beta(2, 15, num_shots_to_simulate) * 0.8 # Simulate xG (low values more likely)
    })
    # Increase xG slightly for goals
    sim_shots.loc[sim_shots['outcome'] == "Goal", 'xg_simulated'] *= np.random.uniform(1.5, 3.0, (sim_shots['outcome'] == "Goal").sum())
    sim_shots['xg_simulated'] = sim_shots['xg_simulated'].clip(0.01, 0.99)

    # Calculate summary stats
    goals_scored = (sim_shots['outcome'] == "Goal").sum()
    total_xg = sim_shots['xg_simulated'].sum()
    performance_vs_xg = goals_scored - total_xg

    fig = px.scatter(
        sim_shots,
        x="x",
        y="y",
        color="outcome",
        size="xg_simulated",
        hover_name="player",
        hover_data={"match_id": True, "xg_simulated": ":.2f"},
        title=f"Conceptual Shot Map & xG for {team_name} ({season}) - Simulated Data",
        color_discrete_map={"Goal": "lime", "Saved": "orange", "Miss/Blocked": "crimson"},
        size_max=15,
        opacity=0.7
    )

    # Basic pitch outline (adjust coordinates for typical pitch viz)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="grey")) # Field boundaries (adjust aspect ratio later)
    fig.add_shape(type="rect", x0=83, y0=21, x1=100, y1=79, line=dict(color="grey")) # Penalty Area (approx)
    fig.add_shape(type="rect", x0=94, y0=36, x1=100, y1=64, line=dict(color="grey")) # 6-yard box (approx)
    fig.add_shape(type="circle", x0=88, y0=43, x1=90, y1=57, line=dict(color="grey")) # Penalty spot (approx)
    # Add center circle and halfway line if showing full pitch

    fig.update_layout(
        xaxis=dict(range=[50, 101], showgrid=False, zeroline=False, showticklabels=False, title="Attacking Direction ->"),
        yaxis=dict(range=[0, 101], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=0.65, title="Pitch Width"),
        width=750, height=500, # Adjust for better pitch aspect ratio
        annotations=[
            dict(x=55, y=95, text=f"Goals: {goals_scored}", showarrow=False, align='left'),
            dict(x=55, y=90, text=f"Total xG: {total_xg:.2f}", showarrow=False, align='left'),
            dict(x=55, y=85, text=f"Perf vs xG: {performance_vs_xg:+.2f}", showarrow=False, align='left', font=dict(color="green" if performance_vs_xg > 0 else "red"))
        ]
    )
    return fig

# VISUALIZATION 11: Player Performance Consistency Matrix
def create_player_consistency_matrix(player_attr_df, players_df_local, min_minutes=MIN_MINUTES_PLAYED_CONSISTENCY):
    logging.info("Creating Visualization 11: Player Consistency Matrix")
    # Requires minutes played data per match, which is not available.
    # We will use number of appearances as a proxy and overall_rating for performance.
    logging.warning("Using number of appearances as proxy for minutes played for Vis 11.")

    # Calculate stats per player over all their attribute entries
    consistency_data = player_attr_df.groupby("player_api_id").agg(
        avg_rating=('overall_rating', 'mean'),
        std_dev_rating=('overall_rating', 'std'),
        num_appearances=('id', 'count') # Using count of attribute entries as proxy
    ).reset_index()

    # Merge with player names
    consistency_data = pd.merge(consistency_data, players_df_local[["player_api_id", "player_name", "age"]], on="player_api_id")

    # Filter based on proxy for minutes played
    consistency_data = consistency_data[consistency_data['num_appearances'] > 10] # Need multiple data points for std dev
    consistency_data = consistency_data.dropna(subset=['avg_rating', 'std_dev_rating'])

    # Lower std dev is better (more consistent)
    consistency_data['consistency_score'] = -consistency_data['std_dev_rating'] # Invert for plot

    if consistency_data.empty:
        logging.warning("Not enough player data for consistency matrix.")
        return go.Figure(layout_title_text="Not enough data for Consistency Matrix.")

    fig = px.scatter(consistency_data, x="avg_rating", y="std_dev_rating",
                     size="num_appearances", color="age",
                     hover_name="player_name",
                     hover_data={"age": True, "avg_rating": ":.1f", "std_dev_rating": ":.2f", "num_appearances": True},
                     title="Player Performance Consistency (Based on Overall Rating Fluctuation)",
                     color_continuous_scale=px.colors.sequential.Plasma_r, # Reversed plasma
                     labels={"avg_rating": "Average Overall Rating", "std_dev_rating": "Inconsistency (Std Dev of Rating)", "num_appearances": "Data Points"})

    # Add quadrant lines (median)
    median_rating = consistency_data['avg_rating'].median()
    median_std_dev = consistency_data['std_dev_rating'].median()
    fig.add_vline(x=median_rating, line_dash="dash", line_color="grey")
    fig.add_hline(y=median_std_dev, line_dash="dash", line_color="grey")

    # Annotations for quadrants (adjust based on median values)
    fig.add_annotation(x=median_rating*1.05, y=median_std_dev*0.8, text="High Perf / High Consistency", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
    fig.add_annotation(x=median_rating*0.95, y=median_std_dev*0.8, text="Low Perf / High Consistency", showarrow=False, bgcolor="rgba(0,0,255,0.1)")
    fig.add_annotation(x=median_rating*1.05, y=median_std_dev*1.2, text="High Perf / Low Consistency", showarrow=False, bgcolor="rgba(255,165,0,0.1)")
    fig.add_annotation(x=median_rating*0.95, y=median_std_dev*1.2, text="Low Perf / Low Consistency", showarrow=False, bgcolor="rgba(255,0,0,0.1)")

    fig.update_layout(yaxis_title="Inconsistency (Std Dev of Rating - Lower is Better)")
    return fig

# VISUALIZATION 12: Team Chemistry & Passing Network Evolution (Static Comparison)
def create_passing_network_comparison(matches_df_local, players_df_local, team_name, season1="2008/2009", season2="2015/2016"):
    logging.info("Creating Visualization 12: Passing Network Comparison")
    # Static comparison between two seasons due to animation complexity and lack of pass data.
    # Uses the same co-occurrence logic as Vis 8.
    logging.warning("Simulating player network based on co-occurrence for Vis 12.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Network {season1}", f"Network {season2}"))

    for i, season in enumerate([season1, season2]):
        team_matches = matches_df_local[(matches_df_local['season'] == season) & 
                                        ((matches_df_local['home_team_name'] == team_name) | (matches_df_local['away_team_name'] == team_name))]
        if team_matches.empty:
            logging.warning(f"No matches found for {team_name} in {season} for network.")
            continue

        player_cols = [f"{side}_player_{j}" for side in ["home", "away"] for j in range(1, 12)]
        involved_player_ids = pd.unique(team_matches[player_cols].values.ravel('K'))
        involved_player_ids = involved_player_ids[~np.isnan(involved_player_ids)]
        player_info = players_df_local[players_df_local['player_api_id'].isin(involved_player_ids)][['player_api_id', 'player_name']].set_index('player_api_id')

        edges = {}
        for _, row in team_matches.iterrows():
            lineup_ids = [row[col] for col in player_cols if pd.notna(row[col])]
            for k in range(len(lineup_ids)):
                for l_idx in range(k + 1, len(lineup_ids)):
                    p1, p2 = sorted((lineup_ids[k], lineup_ids[l_idx]))
                    if p1 != p2:
                        edges[(p1, p2)] = edges.get((p1, p2), 0) + 1

        if not edges or not involved_player_ids.size > 0:
            logging.warning(f"No player connections found for {team_name} in {season}.")
            continue

        nodes_x = np.random.rand(len(involved_player_ids)) * 100
        nodes_y = np.random.rand(len(involved_player_ids)) * 100
        node_ids = involved_player_ids.tolist()
        node_map = {pid: idx for idx, pid in enumerate(node_ids)}

        edge_x = []
        edge_y = []
        for (p1, p2), weight in edges.items():
            if p1 in node_map and p2 in node_map:
                x0, y0 = nodes_x[node_map[p1]], nodes_y[node_map[p1]]
                x1, y1 = nodes_x[node_map[p2]], nodes_y[node_map[p2]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        node_trace = go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text',
                              marker=dict(size=10, color='orange'),
                              text=[player_info.loc[pid]['player_name'] if pid in player_info.index else f"ID: {int(pid)}" for pid in node_ids],
                              textposition='middle right', textfont=dict(size=8),
                              hoverinfo='text',
                              hovertext=[player_info.loc[pid]['player_name'] if pid in player_info.index else f"ID: {int(pid)}" for pid in node_ids],
                              showlegend=False)

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                              line=dict(width=0.5, color='#888'),
                              hoverinfo='none', showlegend=False)

        fig.add_trace(edge_trace, row=1, col=i+1)
        fig.add_trace(node_trace, row=1, col=i+1)
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i+1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i+1)

    fig.update_layout(title=f'Team Network Comparison for {team_name} ({season1} vs {season2}) - Based on Co-occurrence', height=600)
    return fig


# VISUALIZATION 13: "Giant Killer" / "Flat-Track Bully" Identifier
def create_giant_killer_identifier(matches_df_local, teams_df_local):
    """Identify teams that over-/under-perform against top, mid-table and bottom opponents."""
    logging.info("Creating Visualization 13: Giant Killer Identifier")

    # 1) Compute points per match
    def get_points(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return 3, 0
        if row["home_team_goal"] < row["away_team_goal"]:
            return 0, 3
        return 1, 1

    points = matches_df_local.apply(get_points, axis=1, result_type="expand")
    matches_df_local["home_team_points"] = points[0]
    matches_df_local["away_team_points"] = points[1]

    # 2) Aggregate home and away stats separately
    home = (
        matches_df_local
        .groupby(["season", "home_team_api_id"])
        .agg(home_points_sum=("home_team_points", "sum"),
             home_matches=("home_team_points", "count"))
        .reset_index()
        .rename(columns={"home_team_api_id": "team_api_id"})
    )
    away = (
        matches_df_local
        .groupby(["season", "away_team_api_id"])
        .agg(away_points_sum=("away_team_points", "sum"),
             away_matches=("away_team_points", "count"))
        .reset_index()
        .rename(columns={"away_team_api_id": "team_api_id"})
    )

    # 3) Merge and compute total points & matches
    combined = pd.merge(home, away, on=["season", "team_api_id"], how="outer").fillna(0)
    combined["total_points"] = (
        combined["home_points_sum"] + combined["away_points_sum"]
    )
    combined["total_matches"] = (
        combined["home_matches"] + combined["away_matches"]
    )
    combined["avg_ppg"] = combined["total_points"] / combined["total_matches"]

    # 4) Determine season ranks and classify opponent strength
    season_ranks = (
        combined
        .groupby("season")[["team_api_id", "total_points"]]
        .sum()
        .reset_index()
        .assign(
            rank=lambda df: df.groupby("season")["total_points"]
                               .rank(method="dense", ascending=False)
        )
        .set_index(["season", "team_api_id"])["rank"]
    )

    def classify_row(row, as_home: bool):
        opp_id = (row["away_team_api_id"] if as_home else row["home_team_api_id"])
        opp_rank = season_ranks.get((row["season"], opp_id), np.nan)
        num_teams = season_ranks.loc[row["season"]].index.size
        if np.isnan(opp_rank):
            return "Unknown"
        if opp_rank <= 4:
            return "Top 4"
        if opp_rank >= num_teams - 3:
            return "Bottom 4"
        return "Mid-Table"

    # 5) Build detailed results
    records = []
    for is_home, pts_col, matches_col in (
        (True, "home_team_points", "home_matches"),
        (False, "away_team_points", "away_matches"),
    ):
        col_matches = "home_matches" if is_home else "away_matches"
        for _, match in matches_df_local.iterrows():
            opp_type = classify_row(match, as_home=is_home)
            pts = match[pts_col]
            records.append({
                "team_id": match["home_team_api_id" if is_home else "away_team_api_id"],
                "team_name": match["home_team_name" if is_home else "away_team_name"],
                "opponent_type": opp_type,
                "points": pts,
                "matches": 1,
            })

    detail = pd.DataFrame(records)
    agg = (
        detail
        .groupby(["team_name", "opponent_type"])
        .agg(
            total_points=("points", "sum"),
            total_matches=("matches", "sum"),
        )
        .reset_index()
    )
    agg["avg_ppg"] = agg["total_points"] / agg["total_matches"]

    # 6) Filter and plot top performers
    agg = agg[agg["total_matches"] > 20]
    top_teams = (
        agg.groupby("team_name")["avg_ppg"]
           .mean()
           .nlargest(15)
           .index
           .tolist()
    )
    plot_df = agg[agg["team_name"].isin(top_teams)]

    fig = px.bar(
        plot_df,
        x="team_name",
        y="avg_ppg",
        color="opponent_type",
        barmode="group",
        title="Team Performance vs. Opponent Strength (Avg PPG)",
        labels={
            "avg_ppg": "Average Points Per Game",
            "team_name": "Team",
            "opponent_type": "Opponent Tier",
        },
        category_orders={"opponent_type": ["Top 4", "Mid-Table", "Bottom 4"]},
    )
    fig.update_layout(xaxis_tickangle=45, height=600)
    return fig

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    # Load and preprocess data
    players_raw, player_attributes_raw, teams_raw, team_attributes_raw, matches_raw, leagues_raw, countries_raw = load_data()
    players_df, player_attributes_df, teams_df, team_attributes_df, matches_df = preprocess_data(
        players_raw, player_attributes_raw, teams_raw, team_attributes_raw, matches_raw, leagues_raw, countries_raw
    )

    # Generate and save visualizations
    visualizations_to_run = {
        1: create_player_potential_matrix,
        2: create_player_attributes_radar,
        3: create_team_performance_analyzer,
        4: create_possession_outcome_explorer,
        5: create_player_development_trajectories,
        6: create_player_role_fingerprint,
        7: create_team_tactical_matchup_matrix,
        8: create_player_contribution_network,
        9: create_league_competitiveness_dashboard,
        10: create_shot_map_conceptual,
        11: create_player_consistency_matrix,
        12: create_passing_network_comparison,
        13: create_giant_killer_identifier
    }

    filenames = {
        1: "player_potential_matrix.html",
        2: "player_attributes_radar.html",
        3: "team_performance_trajectory.html",
        4: "possession_outcome_explorer.html",
        5: "player_development_trajectories.html",
        6: "player_role_fingerprint.html",
        7: "team_tactical_matchup_matrix.html",
        8: "player_contribution_network.html",
        9: "league_competitiveness_dashboard.html",
        10: "shot_map_conceptual.html",
        11: "player_consistency_matrix.html",
        12: "passing_network_comparison.html",
        13: "giant_killer_identifier.html"
    }

    # Default arguments (can be customized)
    default_args = {
        2: {"df": players_df, "player_names_list": ["Lionel Messi", "Cristiano Ronaldo", "Wayne Rooney"]},
        3: {"matches_df_local": matches_df, "teams_df_local": teams_df, "league_name": "England Premier League"},
        5: {"player_attr_df": player_attributes_df, "players_df_local": players_df},
        6: {"df": players_df, "player_name": "Lionel Messi"},
        7: {"matches_df_local": matches_df, "teams_merged_df_local": teams_df},
        8: {"matches_df_local": matches_df, "players_df_local": players_df, "team_name": "FC Barcelona", "season": "2015/2016"},
        9: {"matches_df_local": matches_df, "teams_df_local": teams_df, "league_name": "England Premier League"},
        10: {"matches_df_local": matches_df, "team_name": "Arsenal", "season": "2015/2016"},
        11: {"player_attr_df": player_attributes_df, "players_df_local": players_df},
        12: {"matches_df_local": matches_df, "players_df_local": players_df, "team_name": "Manchester United", "season1": "2008/2009", "season2": "2015/2016"},
        13: {"matches_df_local": matches_df, "teams_df_local": teams_df}
    }

    for i, func in visualizations_to_run.items():
        try:
            logging.info(f"--- Generating Visualization {i} --- ")
            args = default_args.get(i, {})
            # Default arg for functions taking only players_df
            if i in [1]:
                 args = {"df": players_df}
            # Default arg for functions taking only matches_df
            if i in [4]:
                 args = {"df": matches_df}

            fig = func(**args)
            if fig:
                output_path = os.path.join(OUTPUT_DIR, filenames[i])
                fig.write_html(output_path)
                logging.info(f"Visualization {i} saved to {output_path}")
            else:
                logging.warning(f"Visualization {i} did not return a figure object.")
        except Exception as e:
            logging.error(f"Error generating visualization {i}: {e}", exc_info=True)

    logging.info("--- All visualizations generated (or attempted) --- ")

