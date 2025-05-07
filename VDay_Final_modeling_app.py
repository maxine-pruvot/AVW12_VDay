#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:48:36 2025

@author: maxine
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from scipy.spatial import cKDTree

# MODEL PARAMETERS
#here they are defined but some will be changed by different cursors so the values will change
params = {
    'phi': 1,           # Infection probability when touched (0-1)
    'latent_period': 10,  # Frames until infected cells burst
    'beta': 8,            # Burst size (virions per cell)
    'dS': 0.005,          # Death rate for healthy cells
    'dI': 0.00,           # Death rate for infected cells
    'dV': 0.01,           # Decay rate for viruses
    'mu': 0.05,           # Reproduction rate
    'K': 50,              # Carrying capacity
    'infection_radius': 1.0,   # Interaction radius (cell size)
    'world_size': (500, 500),
    'max_speed': 4        # Movement speed
}

class SIVSimulation:
    def __init__(self, params):
        self.params = params
        # Initial populations
        self.S = self._create_population(params['host_count'], 'lime', 300) #initial number, color
        self.I = self._create_population(0, 'darkorange', 300)  
        self.V = self._create_population(params['virus_count'], 'red', 5)
        self.infected_times = []

    def _create_population(self, n, color, size):
        return {
            'pos': np.random.rand(n, 2) * self.params['world_size'],
            'vel': np.random.uniform(-self.params['max_speed'], self.params['max_speed'], (n, 2)),
            'color': color,
            'size': size
        }

    def update(self):
        # Movement with boundary handling
        for pop in [self.S, self.I, self.V]:
            pop['pos'] += pop['vel']
            # Reflect off walls
            for dim in [0, 1]:
                mask_low = pop['pos'][:, dim] <= 0
                mask_high = pop['pos'][:, dim] >= self.params['world_size'][dim]
                pop['vel'][mask_low, dim] = np.abs(pop['vel'][mask_low, dim])
                pop['vel'][mask_high, dim] = -np.abs(pop['vel'][mask_high, dim])
            pop['pos'] = np.clip(pop['pos'], 0, self.params['world_size'])

        # Infection process (V -> S)
        if len(self.S['pos']) > 0 and len(self.V['pos']) > 0: #checks if there are S and V alive
            tree = cKDTree(self.S['pos']) # Uses a k-D tree to efficiently find nearest neighbors
            dists, s_indices = tree.query(self.V['pos'], k=1) # Finds the nearest susceptible cell to each virus. dists contains the distances and s_indices contains the index of cell
            
            # Find all viruses that could infect
            potential_infections = np.where(dists < params['infection_radius'])[0] # Find viruses within infection radius, array of indexes
            np.random.shuffle(potential_infections) # Randomize infection order to make the infection fair and independent to how agents are stored in the array
            
            for v_idx in potential_infections: #loop on agents in the infection radius, meaning (if they touch...)
                if np.random.rand() < params['phi']: #probability of adsorption (here it is 1)
                    s_idx = s_indices[v_idx] #index of a susceptible cell withing range
                    
                    # Convert S to I and record infection time
                    self.I['pos'] = np.vstack([self.I['pos'], self.S['pos'][s_idx]]) #the position is now in the "infected list"
                    self.I['vel'] = np.vstack([self.I['vel'], self.S['vel'][s_idx]]) #the velocity is now in the "infected list"
                    self.infected_times.append(0) #start the counter for latent period
                    
                    # Remove the infected susceptible cell
                    self.S['pos'] = np.delete(self.S['pos'], s_idx, axis=0) 
                    self.S['vel'] = np.delete(self.S['vel'], s_idx, axis=0)
                    
                    # Remove the infecting virus, the virus infect only once
                    self.V['pos'] = np.delete(self.V['pos'], v_idx, axis=0)
                    self.V['vel'] = np.delete(self.V['vel'], v_idx, axis=0)
                    break

        # Bursting of infected cells
        new_viruses = []
        new_virus_vels = []
        keep_mask = np.ones(len(self.I['pos']), dtype=bool)
        for i in range(len(self.infected_times)):
            self.infected_times[i] += 1
            if self.infected_times[i] >= self.params['latent_period']: #burst if: time of infection > latent period
                # Create new viruses
                burst_pos = self.I['pos'][i] #the position of burst
                new_viruses.extend([burst_pos] * self.params['beta']) #beta new virus in the position of the burst
                new_virus_vels.extend(
                    np.random.uniform(-self.params['max_speed'],
                                      self.params['max_speed'],
                                      (self.params['beta'], 2)) #they are all assigned new velocities
                )

                keep_mask[i] = False

        if len(new_viruses) > 0:
            self.V['pos'] = np.vstack([self.V['pos'], new_viruses])
            self.V['vel'] = np.vstack([self.V['vel'], new_virus_vels])
            self.I['pos'] = self.I['pos'][keep_mask] 
            self.I['vel'] = self.I['vel'][keep_mask]
            self.infected_times = [t for i, t in enumerate(self.infected_times) if keep_mask[i]] #update the list of infection times

        # Death processes
        dt = 0.3
        for pop, rate in [(self.S, self.params['dS']),
                           (self.I, self.params['dI'])]:
            if len(pop['pos']) > 0:
                death_prob = 1 - np.exp(-rate * dt) #function of the death probability
                death_mask = np.random.rand(len(pop['pos'])) < death_prob
                pop['pos'] = pop['pos'][~death_mask]
                pop['vel'] = pop['vel'][~death_mask]
                if pop is self.I:
                    self.infected_times = [t for i, t in enumerate(self.infected_times) if not death_mask[i]]

        # Virus decay
        if len(self.V['pos']) > 0:
            death_prob = 1 - np.exp(-self.params['dV'] * dt) #function of th eviral decay probability
            death_mask = np.random.rand(len(self.V['pos'])) < death_prob
            self.V['pos'] = self.V['pos'][~death_mask]
            self.V['vel'] = self.V['vel'][~death_mask]
        
        # Cell reproduction
        if len(self.S['pos']) > 0 and len(self.S['pos']) < self.params['K']:
            repro_prob = self.params['mu'] * (1 - len(self.S['pos']) / self.params['K'])
            repro_mask = np.random.rand(len(self.S['pos'])) < repro_prob
            if np.any(repro_mask):
                offspring_pos = self.S['pos'][repro_mask] + np.random.normal(0, 0.5, (np.sum(repro_mask), 2))
                offspring_vel = np.random.uniform(-self.params['max_speed'], self.params['max_speed'],
                                                    (np.sum(repro_mask), 2))
                self.S['pos'] = np.vstack([self.S['pos'], offspring_pos])
                self.S['vel'] = np.vstack([self.S['vel'], offspring_vel])

def run_simulation(params_run):
    params.update({
        'host_count': params_run['host_count'],
        'virus_count': params_run['virus_count'],
        'beta': params_run['burst_size'],
        'latent_period': params_run['latent_period'],
        'dV': params_run['virus_decay_rate'],
        'mu': params_run['growth_rate'],
        'max_speed': params_run['speed'],
        'phi': params_run['infection_prob'],
        'infection_radius': params_run['infection_radius'],
        'world_size': (200, 200)
    })

    sim = SIVSimulation(params)
    plot_placeholder = st.empty()
    population_placeholder = st.empty()
    time_placeholder = st.empty()

    start_time = time.time()

    # Init figure only ONCE
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, params['world_size'][0])
    ax.set_ylim(0, params['world_size'][1])
    ax.axis('off')

    scat_I = ax.scatter([], [], c='darkorange', s=300, alpha=0.9, edgecolors='black')
    scat_S = ax.scatter([], [], c='lime', s=300, alpha=0.9, edgecolors='black')
    scat_V = ax.scatter([], [], c='red', s=5, alpha=0.7)

    # No need to add legend every frame
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Cellules saines'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=10, label='Cellules infectées'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Virus')
    ], loc='upper right')

    S_history, I_history, V_history = [], [], []

    for frame in range(10000):
        elapsed = time.time() - start_time
        time_placeholder.markdown(f"⏱ Temps : {str(timedelta(seconds=int(elapsed)))}")

        sim.update()

        # Update scatter points
        scat_S.set_offsets(sim.S['pos'] if len(sim.S['pos']) > 0 else np.empty((0, 2)))
        scat_I.set_offsets(sim.I['pos'] if len(sim.I['pos']) > 0 else np.empty((0, 2)))
        scat_V.set_offsets(sim.V['pos'] if len(sim.V['pos']) > 0 else np.empty((0, 2)))

        # Redraw the figure
        plot_placeholder.pyplot(fig, clear_figure=False)  # 👈🏼 Don't clear figure each time

        # Store history
        S_history.append(len(sim.S['pos']))
        I_history.append(len(sim.I['pos']))
        V_history.append(len(sim.V['pos']))

        # Only update the population curve every 20 frames to avoid lag
        if frame % 20 == 0:
            fig_population, ax_population = plt.subplots(figsize=(6, 3))
            ax_population.plot(S_history, label='Cellules saines', color='lime')
            ax_population.plot(I_history, label='Cellules infectées', color='darkorange')
            ax_population.plot(V_history, label='Virus', color='red')
            ax_population.set_xlabel('Temps')
            ax_population.set_ylabel('Population')
            ax_population.set_yscale('log')
            ax_population.legend()
            population_placeholder.pyplot(fig_population)
            plt.close(fig_population)

        # Check termination conditions
        if len(sim.S['pos']) == 0 and len(sim.I['pos']) == 0 and len(sim.V['pos']) == 0:
            st.warning("Toutes les cellules et les virus ont disparu !")
            break
        if len(sim.S['pos']) > 0 and len(sim.I['pos']) == 0 and len(sim.V['pos']) == 0:
            break
        
    plt.close(fig)


def main():
    st.title("🔬 Processus d'infection")

    # Inject custom CSS for larger sidebar text
    st.markdown("""
        <style>
        [data-testid="stSidebar"] * {
            font-size: 22px !important;
        }
        .stMarkdown {
            font-size: 22px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        params_run = {
            'host_count': st.slider("🦠 Nombre de cellules saines", 1, 40, 10), #st.slider(name, min vlaue, max vlaue, starting value)
            'virus_count': st.slider("🔻 Nombre de virus", 10, 200, 100),
            'infection_prob': 1.0,
            'growth_rate': st.slider("📈🦠 Taux de croissance des cellules saines", 0.02, 0.06, 0.03, 0.01),
            'latent_period': st.slider("⏱️💥 Temps de latence", 10, 60, 30),
            'burst_size': st.slider("💥🔻 Nombre de virus libérés par cellule infectée", 1, 50, 8),
            'virus_decay_rate': 0.08, #st.slider("☠️🔻 Taux de mort des virus", 0.0, 0.02, 0.01, 0.005),
            'speed': 1,
            'infection_radius': 5 # manual way to have the infection when the virus touches the membrane of the cell
        }

    if st.button("▶️ Commencer la simulation"):
        run_simulation(params_run)


if __name__ == "__main__":
    main()
